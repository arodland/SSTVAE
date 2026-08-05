#include "modem/diversity.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <variant>

#include "framing/framing.hpp"
#include "latents/latents.hpp"

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

// Raw-array form, so it works identically whether the branches are
// DemodResult, BlindDemodResult, or padded copies of either -- shared
// by every combine_*/branch_contribution below.
MrcWeights mrc_weights_raw(const std::vector<const std::vector<double>*>& weights_ptrs,
                          const std::vector<double>& snr_dbs) {
    const std::size_t nb = weights_ptrs.size();
    const std::size_t nl = weights_ptrs[0]->size();
    std::vector<double> snr_lin(nb);
    bool any_positive = false;
    for (std::size_t i = 0; i < nb; ++i) {
        const double s = snr_dbs[i];
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
        const std::vector<double>& wt = *weights_ptrs[i];
        for (std::size_t k = 0; k < nl; ++k) {
            w[i * nl + k] = s * wt[k] * wt[k];
        }
    }
    return {std::move(snr_lin), std::move(w), nl};
}

MrcWeights mrc_weights(const std::vector<DemodResult>& results) {
    std::vector<const std::vector<double>*> w_ptrs;
    std::vector<double> snr_dbs;
    for (const DemodResult& r : results) {
        w_ptrs.push_back(&r.weights);
        snr_dbs.push_back(r.snr_db);
    }
    return mrc_weights_raw(w_ptrs, snr_dbs);
}

struct CombinedArrays {
    std::vector<double> latents;
    std::vector<double> weights;
    double snr_db;
};

// The core MRC arithmetic shared by every combine_* function below, over
// raw arrays rather than any particular result type. Needs at least 2
// branches; callers handle the single-branch identity case themselves.
CombinedArrays mrc_combine_arrays(const std::vector<const std::vector<double>*>& latents_ptrs,
                                  const std::vector<const std::vector<double>*>& weights_ptrs,
                                  const std::vector<double>& snr_dbs) {
    const MrcWeights mw = mrc_weights_raw(weights_ptrs, snr_dbs);
    const std::size_t nb = latents_ptrs.size();
    const std::size_t nl = mw.n_latents;
    const double ref = *std::max_element(mw.snr_lin.begin(), mw.snr_lin.end());

    std::vector<double> latents(nl, 0.0), weights(nl, 0.0);
    for (std::size_t k = 0; k < nl; ++k) {
        double num = 0.0, denom = 0.0;
        for (std::size_t i = 0; i < nb; ++i) {
            const double wk = mw.w[i * nl + k];
            num += wk * (*latents_ptrs[i])[k];
            denom += wk;
        }
        if (denom > 0.0) {
            latents[k] = num / denom;
            weights[k] = std::min(1.0, std::sqrt(denom / ref));
        }
    }
    double total_snr_lin = 0.0;
    for (double s : mw.snr_lin) total_snr_lin += s;
    return {std::move(latents), std::move(weights), 10.0 * std::log10(total_snr_lin)};
}

// Index of the branch with the highest finite snr_db (ties keep the
// first, matching std::max_element/Python's max() with a custom key).
template <typename T, typename SnrOf>
std::size_t best_branch(const std::vector<T>& results, SnrOf snr_of) {
    std::size_t best_i = 0;
    double best_snr = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < results.size(); ++i) {
        const double s = snr_of(results[i]);
        if (std::isfinite(s) && s > best_snr) {
            best_snr = s;
            best_i = i;
        }
    }
    return best_i;
}

double snr_of_demod(const DemodResult& r) { return r.snr_db; }
double snr_of_blind(const BlindDemodResult& r) { return r.snr_db; }

// Per (row, frame) cell, averaged over whichever latents this frame's
// slots bucket into that row -- shared by both contribution_image
// overloads. `row_of(position_in_frame_slots)` decides what a slot's
// row is; contribution_image passes the carrier-index rule. `overall`
// is the branches' combined confidence at every latent (see
// combined_weight below), used to scale brightness relative to this
// reception's own peak cell, on top of frac's fractional-share hue.
images::Picture render_contribution_grid(
    const std::vector<std::vector<double>>& frac, const std::vector<double>& overall,
    int n_frames, int n_rows, int scale, const std::function<int(std::size_t)>& row_of) {
    std::vector<double> sum_r(static_cast<std::size_t>(n_rows) * n_frames, 0.0);
    std::vector<double> sum_b(sum_r.size(), 0.0);
    std::vector<double> sum_w(sum_r.size(), 0.0);
    std::vector<int> counts(sum_r.size(), 0);

    for (int f = 0; f < n_frames; ++f) {
        const framing::FrameSlots fs = framing::slot_range_for_frame(f);
        for (std::size_t pos = 0; pos < fs.indices.size(); ++pos) {
            const std::int64_t idx = fs.indices[pos];
            const int row = row_of(pos);
            const std::size_t cell = static_cast<std::size_t>(row) * n_frames + f;
            sum_r[cell] += frac[0][static_cast<std::size_t>(idx)];
            sum_b[cell] += frac[1][static_cast<std::size_t>(idx)];
            sum_w[cell] += overall[static_cast<std::size_t>(idx)];
            ++counts[cell];
        }
    }

    // Per-cell means first, then find the reception's own peak cell --
    // brightness is normalized to *that*, not to the raw [0, 1]
    // confidence scale (see the header comment for why).
    std::vector<double> mean_r(sum_r.size(), 0.0), mean_b(sum_r.size(), 0.0),
        mean_w(sum_r.size(), 0.0);
    double peak = 0.0;
    for (std::size_t cell = 0; cell < sum_r.size(); ++cell) {
        if (counts[cell] == 0) continue;
        mean_r[cell] = sum_r[cell] / counts[cell];
        mean_b[cell] = sum_b[cell] / counts[cell];
        mean_w[cell] = sum_w[cell] / counts[cell];
        peak = std::max(peak, mean_w[cell]);
    }

    const int sc = std::max(1, scale);
    const int out_w = n_frames * sc;
    const int out_h = n_rows * sc;
    images::Picture img(out_w, out_h);

    // Nearest-neighbor replication by construction, not a smoothing
    // resize -- each cell is a categorical (row, frame) value, and
    // stb's srgb resizer (images::resize) would blur cell boundaries
    // that are meaningful, not noise.
    for (int row = 0; row < n_rows; ++row) {
        for (int f = 0; f < n_frames; ++f) {
            const std::size_t cell = static_cast<std::size_t>(row) * n_frames + f;
            std::uint8_t r = 0, b = 0;
            if (counts[cell] > 0 && peak > 0.0) {
                const double norm = mean_w[cell] / peak;
                r = static_cast<std::uint8_t>(
                    std::clamp(mean_r[cell] * norm * 255.0, 0.0, 255.0));
                b = static_cast<std::uint8_t>(
                    std::clamp(mean_b[cell] * norm * 255.0, 0.0, 255.0));
            }
            for (int dy = 0; dy < sc; ++dy) {
                const int y = row * sc + dy;
                for (int dx = 0; dx < sc; ++dx) {
                    const int x = f * sc + dx;
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

// A transmitted latent's *position* within one frame's slots (not the
// canonical index at that position, which the interleaver's
// PAPR-motivated permutation scatters) encodes its OFDM carrier:
// slots_to_symbols reshapes a frame's LATENTS_PER_FRAME slots as
// (DATA_SYMS_PER_FRAME, NC_LATENT, 2), so position k's carrier is
// (k / 2) % NC_LATENT -- real/imag-independent and identical across
// every group and mode, unlike the decoder channel idx[k] happens to
// land on.
int carrier_of_position(std::size_t pos) {
    return static_cast<int>((pos / 2) % static_cast<std::size_t>(config::NC_LATENT));
}

// frac: branch_contribution's (n_branches, n_latents) fractional share.
// overall: (n_latents,) the branches' combined confidence -- the same
// min(sqrt(sum(mrc_w)/ref), 1) mrc_combine_arrays reports as a
// DemodResult's post-combine weights. One pass computes both, since
// they share the same MrcWeights -- branch_contribution below returns
// just .frac; contribution_image needs both, to scale hue by overall
// strength.
struct ContributionData {
    std::vector<std::vector<double>> frac;
    std::vector<double> overall;
};

ContributionData contribution_data(const std::vector<DemodResult>& results) {
    validate_branches(results);
    const std::size_t nb = results.size();
    const std::size_t nl = results[0].latents.size();
    std::vector<std::vector<double>> frac(nb, std::vector<double>(nl, 0.0));
    std::vector<double> overall(nl, 0.0);

    if (nb == 1) {
        for (std::size_t k = 0; k < nl; ++k) {
            frac[0][k] = results[0].weights[k] > 0.0 ? 1.0 : 0.0;
            overall[k] = results[0].weights[k];
        }
        return {std::move(frac), std::move(overall)};
    }

    const MrcWeights mw = mrc_weights(results);
    const double ref = *std::max_element(mw.snr_lin.begin(), mw.snr_lin.end());
    for (std::size_t k = 0; k < nl; ++k) {
        double denom = 0.0;
        for (std::size_t i = 0; i < nb; ++i) denom += mw.w[i * nl + k];
        if (denom > 0.0) {
            for (std::size_t i = 0; i < nb; ++i) frac[i][k] = mw.w[i * nl + k] / denom;
            overall[k] = std::min(1.0, std::sqrt(denom / ref));
        }
    }
    return {std::move(frac), std::move(overall)};
}

ContributionData contribution_data(const std::vector<Branch>& results) {
    if (results.empty()) throw std::invalid_argument("branch_contribution needs at least one branch");
    if (results.size() == 1) {
        const std::vector<double>* w = nullptr;
        if (const auto* d = std::get_if<DemodResult>(&results[0])) {
            w = &d->weights;
        } else {
            w = &std::get<BlindDemodResult>(results[0]).weights;
        }
        std::vector<double> frac0(w->size()), overall(w->size());
        for (std::size_t k = 0; k < w->size(); ++k) {
            frac0[k] = (*w)[k] > 0.0 ? 1.0 : 0.0;
            overall[k] = (*w)[k];
        }
        return {{std::move(frac0)}, std::move(overall)};
    }

    std::vector<DemodResult> headered;
    std::vector<BlindDemodResult> blind;
    for (const Branch& b : results) {
        if (const auto* d = std::get_if<DemodResult>(&b)) headered.push_back(*d);
        else blind.push_back(std::get<BlindDemodResult>(b));
    }

    std::vector<std::vector<double>> padded;  // owns padded copies, if any
    std::vector<const std::vector<double>*> w_ptrs;
    std::vector<double> snr_dbs;
    if (blind.empty()) {
        validate_branches(headered);
        for (const DemodResult& r : headered) {
            w_ptrs.push_back(&r.weights);
            snr_dbs.push_back(r.snr_db);
        }
    } else {
        if (headered.size() > 1) validate_branches(headered);
        padded.reserve(headered.size());
        for (const DemodResult& r : headered) padded.push_back(latents::pad_to_full(r.weights));
        std::size_t next_padded = 0;
        for (const Branch& b : results) {
            if (const auto* d = std::get_if<DemodResult>(&b)) {
                w_ptrs.push_back(&padded[next_padded++]);
                snr_dbs.push_back(d->snr_db);
            } else {
                const auto& bd = std::get<BlindDemodResult>(b);
                w_ptrs.push_back(&bd.weights);
                snr_dbs.push_back(bd.snr_db);
            }
        }
    }

    const MrcWeights mw = mrc_weights_raw(w_ptrs, snr_dbs);
    const double ref = *std::max_element(mw.snr_lin.begin(), mw.snr_lin.end());
    const std::size_t nb = w_ptrs.size();
    const std::size_t nl = mw.n_latents;
    std::vector<std::vector<double>> frac(nb, std::vector<double>(nl, 0.0));
    std::vector<double> overall(nl, 0.0);
    for (std::size_t k = 0; k < nl; ++k) {
        double denom = 0.0;
        for (std::size_t i = 0; i < nb; ++i) denom += mw.w[i * nl + k];
        if (denom > 0.0) {
            for (std::size_t i = 0; i < nb; ++i) frac[i][k] = mw.w[i * nl + k] / denom;
            overall[k] = std::min(1.0, std::sqrt(denom / ref));
        }
    }
    return {std::move(frac), std::move(overall)};
}

}  // namespace

DemodResult combine_demod_results(const std::vector<DemodResult>& results) {
    if (results.empty())
        throw std::invalid_argument("combine_demod_results needs at least one branch");
    if (results.size() == 1) return results[0];
    validate_branches(results);

    std::vector<const std::vector<double>*> lat_ptrs, w_ptrs;
    std::vector<double> snr_dbs;
    for (const DemodResult& r : results) {
        lat_ptrs.push_back(&r.latents);
        w_ptrs.push_back(&r.weights);
        snr_dbs.push_back(r.snr_db);
    }
    CombinedArrays c = mrc_combine_arrays(lat_ptrs, w_ptrs, snr_dbs);

    const std::size_t best_i = best_branch(results, snr_of_demod);
    int frames_received = results[0].frames_received;
    for (const DemodResult& r : results)
        frames_received = std::max(frames_received, r.frames_received);

    DemodResult out = results[best_i];
    out.latents = std::move(c.latents);
    out.weights = std::move(c.weights);
    out.frames_received = frames_received;
    out.snr_db = c.snr_db;
    return out;
}

BlindDemodResult combine_blind_results(const std::vector<BlindDemodResult>& results) {
    if (results.empty())
        throw std::invalid_argument("combine_blind_results needs at least one branch");
    if (results.size() == 1) return results[0];

    std::vector<const std::vector<double>*> lat_ptrs, w_ptrs;
    std::vector<double> snr_dbs;
    for (const BlindDemodResult& r : results) {
        lat_ptrs.push_back(&r.latents);
        w_ptrs.push_back(&r.weights);
        snr_dbs.push_back(r.snr_db);
    }
    CombinedArrays c = mrc_combine_arrays(lat_ptrs, w_ptrs, snr_dbs);

    const std::size_t best_i = best_branch(results, snr_of_blind);
    int n_frames = results[0].n_frames;
    for (const BlindDemodResult& r : results) n_frames = std::max(n_frames, r.n_frames);

    BlindDemodResult out = results[best_i];
    out.latents = std::move(c.latents);
    out.weights = std::move(c.weights);
    out.n_frames = n_frames;
    out.snr_db = c.snr_db;
    return out;
}

Branch combine_diversity_results(const std::vector<Branch>& results) {
    if (results.empty())
        throw std::invalid_argument("combine_diversity_results needs at least one branch");
    std::vector<DemodResult> headered;
    std::vector<BlindDemodResult> blind;
    for (const Branch& b : results) {
        if (const auto* d = std::get_if<DemodResult>(&b)) headered.push_back(*d);
        else blind.push_back(std::get<BlindDemodResult>(b));
    }

    if (blind.empty()) return combine_demod_results(headered);
    if (headered.empty()) return combine_blind_results(blind);

    if (headered.size() > 1) validate_branches(headered);
    const config::ModeSpec spec = headered[0].mode;
    const auto n = static_cast<std::size_t>(spec.n_latents);

    // Pad every header branch's arrays up to mode C's full size (the
    // blind branches already are that size) so one MRC pass covers all
    // of them, then truncate back down to the header-locked mode's
    // range -- the header is authoritative for what was actually sent.
    std::vector<std::vector<double>> padded_lat, padded_w;
    padded_lat.reserve(headered.size());
    padded_w.reserve(headered.size());
    for (const DemodResult& r : headered) {
        padded_lat.push_back(latents::pad_to_full(r.latents));
        padded_w.push_back(latents::pad_to_full(r.weights));
    }

    std::vector<const std::vector<double>*> lat_ptrs, w_ptrs;
    std::vector<double> snr_dbs;
    for (std::size_t i = 0; i < headered.size(); ++i) {
        lat_ptrs.push_back(&padded_lat[i]);
        w_ptrs.push_back(&padded_w[i]);
        snr_dbs.push_back(headered[i].snr_db);
    }
    for (const BlindDemodResult& r : blind) {
        lat_ptrs.push_back(&r.latents);
        w_ptrs.push_back(&r.weights);
        snr_dbs.push_back(r.snr_db);
    }
    CombinedArrays c = mrc_combine_arrays(lat_ptrs, w_ptrs, snr_dbs);

    const std::size_t best_i = best_branch(headered, snr_of_demod);
    int frames_received = headered[0].frames_received;
    for (const DemodResult& r : headered)
        frames_received = std::max(frames_received, r.frames_received);

    DemodResult out = headered[best_i];
    out.latents.assign(c.latents.begin(), c.latents.begin() + static_cast<std::ptrdiff_t>(n));
    out.weights.assign(c.weights.begin(), c.weights.begin() + static_cast<std::ptrdiff_t>(n));
    out.frames_received = frames_received;
    out.snr_db = c.snr_db;
    return out;
}

std::vector<std::vector<double>> branch_contribution(
    const std::vector<DemodResult>& results) {
    return contribution_data(results).frac;
}

std::vector<std::vector<double>> branch_contribution(const std::vector<Branch>& results) {
    return contribution_data(results).frac;
}

images::Picture contribution_image(const std::vector<DemodResult>& results, int scale) {
    if (results.size() != 2)
        throw std::invalid_argument("contribution_image needs exactly two branches");
    validate_branches(results);
    const ContributionData cd = contribution_data(results);
    return render_contribution_grid(cd.frac, cd.overall, results[0].mode.n_frames,
                                    config::NC_LATENT, scale, carrier_of_position);
}

images::Picture contribution_image(const std::vector<Branch>& results, int scale) {
    if (results.size() != 2)
        throw std::invalid_argument("contribution_image needs exactly two branches");
    const ContributionData cd = contribution_data(results);

    int n_frames;
    if (std::holds_alternative<DemodResult>(results[0]) &&
        std::holds_alternative<DemodResult>(results[1])) {
        const auto& a = std::get<DemodResult>(results[0]);
        const auto& b = std::get<DemodResult>(results[1]);
        if (a.mode.name != b.mode.name) throw std::invalid_argument("branch mode mismatch");
        n_frames = a.mode.n_frames;
    } else {
        n_frames = config::LATENT_GROUPS * config::FRAMES_PER_GROUP;
    }
    return render_contribution_grid(cd.frac, cd.overall, n_frames, config::NC_LATENT, scale,
                                    carrier_of_position);
}

}  // namespace sstvae::modem::diversity
