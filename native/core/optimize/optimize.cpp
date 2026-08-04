#include "optimize/optimize.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

#include "latents/latents.hpp"

namespace sstvae::optimize {

namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

// Back onto the unit-RMS shell, over the transmitted groups only.
//
// That normalization is the on-air contract between encoder, modem and
// training, not a training detail -- and `Modem::modulate` normalizes
// over the *truncated* vector, so a mode A/B optimization that
// normalized across all 132 channels would be solving a different
// problem than the radio poses.
void project_unit_rms(std::vector<float>& z, std::size_t active_len) {
    double sum = 0.0;
    for (std::size_t i = 0; i < active_len; ++i) {
        sum += static_cast<double>(z[i]) * z[i];
    }
    const double rms = std::sqrt(sum / static_cast<double>(active_len));
    const float scale = static_cast<float>(1.0 / std::max(rms, 1e-6));
    for (float& v : z) v *= scale;
}

}  // namespace

std::string to_string(StopReason reason) {
    switch (reason) {
        case StopReason::Plateau: return "plateau";
        case StopReason::TimeBudget: return "time budget";
        case StopReason::MaxSteps: return "max steps";
        case StopReason::Cancelled: return "cancelled";
    }
    return "?";
}

double Result::objective_gain_db() const {
    if (mse_best <= 0.0 || mse_start <= 0.0) return 0.0;
    return 10.0 * std::log10(mse_start / mse_best);
}

Result run(const GradFn& grad_fn, const std::vector<double>& latents,
           const config::ModeSpec& mode, const Options& opts,
           const ProgressFn& progress) {
    if (!grad_fn) throw std::invalid_argument("optimize::run needs a gradient function");
    const std::size_t n_mode = static_cast<std::size_t>(mode.n_latents);
    if (latents.size() < n_mode) {
        throw std::invalid_argument("optimize::run: latent vector shorter than the mode");
    }
    if (opts.channel_samples < 1) {
        throw std::invalid_argument("optimize::run: channel_samples must be >= 1");
    }

    const std::size_t n_full = static_cast<std::size_t>(latents::N_LATENTS);
    // The graph is mode C shaped, always: a shorter mode is a
    // full-length vector whose tail is weighted to zero.
    const std::size_t active_len =
        static_cast<std::size_t>(mode.groups) * config::GROUP_LATENTS;

    // The untransmitted tail is zero here and stays zero: `weights` is
    // zero there, so the gradient is too, and Adam's update for a
    // coordinate whose m and v are both zero is exactly zero. No
    // explicit re-zeroing is needed inside the loop -- a version that
    // did it was unobservable, which is how it survived mutation.
    std::vector<float> z(n_full, 0.0f);
    std::vector<float> weights(n_full, 0.0f);
    for (std::size_t i = 0; i < n_mode; ++i) z[i] = static_cast<float>(latents[i]);
    for (std::size_t i = 0; i < active_len; ++i) weights[i] = 1.0f;
    project_unit_rms(z, active_len);

    std::vector<float> noisy(n_full);
    std::vector<float> grad(n_full);
    std::vector<float> accum(n_full);
    std::vector<float> m(n_full, 0.0f);
    std::vector<float> v(n_full, 0.0f);

    const double sigma = std::pow(10.0, -opts.objective_snr_db / 20.0);
    std::mt19937_64 rng(opts.seed);
    std::normal_distribution<float> gauss(0.0f, static_cast<float>(sigma));

    // Fading draws use a unit normal scaled per latent instead. The flat
    // path deliberately keeps its own `gauss` rather than sharing one
    // unit distribution and multiplying: that would change the numbers
    // the shipping objective produces for no reason at all.
    const FadingProfile* fp = opts.fading_profile;
    std::normal_distribution<float> unit(0.0f, 1.0f);
    std::discrete_distribution<int> bins(
        fp ? fp->prob.begin() : MEASURED_FADING_PROFILE.prob.begin(),
        fp ? fp->prob.end() : MEASURED_FADING_PROFILE.prob.end());
    // Only allocated when fading is on: the flat path must not pay a
    // 634k-float copy per draw for a feature it is not using.
    std::vector<float> faded_weights(fp ? n_full : 0);

    constexpr double b1 = 0.9, b2 = 0.999, eps = 1e-8;

    // Noise-free baseline for `Progress::clean_gain_db`. Taken once,
    // from the encoder's own latents, so every later probe is a gain
    // over the picture the operator would have sent unaided.
    double clean_mse_start = 0.0;
    double clean_gain_db = 0.0;
    const bool probe = opts.clean_probe_every > 0 && progress != nullptr;
    if (probe) {
        double m0 = 0.0;
        grad_fn(z, weights, grad, m0);
        clean_mse_start = m0;
    }

    std::vector<float> best_z = z;
    double best_mse = std::numeric_limits<double>::infinity();
    double mse_start = 0.0;
    int best_step = 0;
    int step = 0;
    StopReason stop = StopReason::MaxSteps;
    const auto t0 = Clock::now();

    for (step = 1; step <= opts.max_steps; ++step) {
        std::fill(accum.begin(), accum.end(), 0.0f);
        double mse_total = 0.0;
        for (int s = 0; s < opts.channel_samples; ++s) {
            // The channel's Jacobian is the identity, so the gradient
            // returned for the *noisy* latents is already the one for
            // z. That is why no channel model has to be ported
            // alongside this: noise in, `weights` back out as the
            // chain-rule factor.
            //
            // A per-latent gain keeps that property: scaling the noise
            // is still an additive perturbation independent of `z`, so
            // the chain rule is unchanged and no second graph run is
            // needed. The drawn confidence goes in as `weights`, which
            // *is* the chain-rule factor -- so a low-confidence latent
            // is both noisier and discounted, exactly as the receiver
            // reports it.
            const std::vector<float>* w_in = &weights;
            if (fp) {
                for (std::size_t i = 0; i < n_full; ++i) {
                    const int b = bins(rng);
                    noisy[i] = z[i] +
                               unit(rng) * static_cast<float>(sigma) * fp->rel_sigma[b];
                    faded_weights[i] = weights[i] * fp->weight[b];
                }
                w_in = &faded_weights;
            } else {
                for (std::size_t i = 0; i < n_full; ++i) noisy[i] = z[i] + gauss(rng);
            }
            double mse = 0.0;
            grad_fn(noisy, *w_in, grad, mse);
            if (grad.size() != n_full) {
                throw std::runtime_error("gradient function returned the wrong size");
            }
            for (std::size_t i = 0; i < n_full; ++i) accum[i] += grad[i];
            mse_total += mse;
        }
        // Averaging the *gradient* is conceptually right and
        // numerically irrelevant: Adam's update is m/sqrt(v), which is
        // invariant to a constant scaling of the gradient, so summing
        // the draws instead would take identical steps. It is kept
        // because the quantity being estimated is an expectation, not
        // because anything downstream can tell. Averaging the **loss**
        // is a different matter -- that number is reported, compared
        // against `mse_best`, and drives the plateau test.
        const float inv_s = 1.0f / static_cast<float>(opts.channel_samples);
        for (std::size_t i = 0; i < n_full; ++i) accum[i] *= inv_s;
        const double mse = mse_total / opts.channel_samples;

        if (step == 1) mse_start = mse;
        if (mse < best_mse * (1.0 - opts.min_rel_gain)) best_step = step;
        if (mse < best_mse) {
            best_mse = mse;
            best_z = z;
        }
        if (progress) {
            Progress p;
            p.step = step;
            p.mse = mse;
            p.best_mse = best_mse;
            p.elapsed_s = seconds_since(t0);
            // Free: both terms are already here, and one log10 a step
            // is nothing against four decoder passes.
            p.objective_gain_db = (best_mse > 0.0 && mse_start > 0.0)
                                      ? 10.0 * std::log10(mse_start / best_mse)
                                      : 0.0;
            // Probe the *current* iterate with the noise switched off.
            // Not `best_z`: the operator is watching this run, and a
            // number that stalls while the loss wanders is harder to
            // read than one that tracks where the search actually is.
            if (probe && (step == 1 || step % opts.clean_probe_every == 0)) {
                double mc = 0.0;
                grad_fn(z, weights, grad, mc);
                if (mc > 0.0 && clean_mse_start > 0.0) {
                    clean_gain_db = 10.0 * std::log10(clean_mse_start / mc);
                }
            }
            p.clean_gain_db = clean_gain_db;
            p.estimated_gain_db = clean_gain_db * RETENTION;
            if (!progress(p)) {
                stop = StopReason::Cancelled;
                break;
            }
        }
        if (step - best_step >= opts.patience) {
            stop = StopReason::Plateau;
            break;
        }
        if (seconds_since(t0) >= opts.time_budget_s) {
            stop = StopReason::TimeBudget;
            break;
        }

        const double bc1 = 1.0 - std::pow(b1, step);
        const double bc2 = 1.0 - std::pow(b2, step);
        for (std::size_t i = 0; i < n_full; ++i) {
            const double g = accum[i];
            m[i] = static_cast<float>(b1 * m[i] + (1.0 - b1) * g);
            v[i] = static_cast<float>(b2 * v[i] + (1.0 - b2) * g * g);
            const double step_i =
                opts.learning_rate * (m[i] / bc1) / (std::sqrt(v[i] / bc2) + eps);
            z[i] = static_cast<float>(z[i] - step_i);
        }
        for (std::size_t i = active_len; i < n_full; ++i) z[i] = 0.0f;
        project_unit_rms(z, active_len);
    }
    if (step > opts.max_steps) step = opts.max_steps;

    Result r;
    r.latents.resize(n_mode);
    for (std::size_t i = 0; i < n_mode; ++i) r.latents[i] = best_z[i];
    r.steps = step;
    r.seconds = seconds_since(t0);
    r.stop = stop;
    r.mse_start = mse_start;
    r.mse_best = best_mse;
    return r;
}

}  // namespace sstvae::optimize
