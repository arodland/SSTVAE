// Transmit-time per-image latent optimization.
//
// The encoder is amortized: one forward pass, trained to minimize loss
// averaged over a dataset. For the particular picture in front of it
// there is generally a better input to the same frozen decoder, and a
// transmission lasts 32-95 s against the encoder's ~30 ms -- so the
// sender can afford to go looking. Measured worth 1.4-1.8 dB of
// *recovered* picture; see docs/latent-optimization.md.
//
// **Sender-side only.** What comes out is an ordinary unit-RMS latent
// vector: same count, same on-air contract, same modem, so every
// existing receiver decodes it with no version negotiation.
//
// The gradient arrives through a `GradFn` seam, for the same reason
// `rx::Engine` takes its decoder as one: everything in this file that
// can be *wrong* -- the Adam step, the unit-RMS projection, the mode
// mask, which iterate is returned, why it stopped -- is arithmetic, and
// keeping it in `sstvae_core` means it builds and is tested with
// `--no-codec`, no onnxruntime and no downloaded artifact. The ORT
// implementation of the seam is `codec/grad_session.hpp`.

#ifndef SSTVAE_OPTIMIZE_OPTIMIZE_HPP
#define SSTVAE_OPTIMIZE_OPTIMIZE_HPP

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "config.hpp"

namespace sstvae::optimize {

// Optimizing against the *clean* decoder is not a milder version of
// this -- it is harmful, losing on every fading channel measured,
// because it finds latents that decode beautifully without noise and
// fall apart with it. The objective runs through a channel model at
// this SNR instead.
//
// 5 dB is measured and the optimum is flat: 2.5-7.5 dB are all within
// 0.2 dB of the peak on two images and two checkpoints, which is why
// this is a constant rather than an operator setting -- a setting would
// have to be guessed before transmitting, and a wrong guess is worse
// than a fixed compromise. The penalty is asymmetric: too *low* costs
// little, too high degrades toward the clean objective. If this ever
// has to move, move it down.
inline constexpr double OBJECTIVE_SNR_DB = 5.0;

// Monte Carlo draws of the channel per step. Lowering it makes each
// step cheaper and noisier -- and weakening the channel term is exactly
// what reopens the failure above, so this is a numerical parameter, not
// a performance knob.
inline constexpr int CHANNEL_SAMPLES = 4;

// Swept 2026-08-02 against the previous 0.02, which was never swept.
// Worth +0.33 dB of recovered picture at 5 steps and +0.37 at 20, end to
// end. Rates above this are faster once moving and unstable starting --
// Adam's first step has magnitude lr exactly, and 0.10 overshoots far
// enough to spend its whole budget recovering. A 2-step warmup into 0.10
// repairs that and won the objective sweep, but by under 0.1 dB end to
// end, which is why this is still a constant in both implementations.
// 0.05 is the largest rate that was never negative on any cell measured.
// See `docs/latent-optimization.md`; keep in step with
// `sstvae/latent_optim.py`, which is normative.
inline constexpr double LEARNING_RATE = 0.05;

// One evaluation of the decoder and its input-gradient.
//
// `latents` and `weights` are mode C shaped (LATENT_CHANNELS * H * W).
// The target picture is bound when the function is made -- it does not
// change across steps. Fills `grad` (same shape) and `mse`.
using GradFn = std::function<void(const std::vector<float>& latents,
                                  const std::vector<float>& weights,
                                  std::vector<float>& grad, double& mse)>;

struct Progress {
    int step = 0;
    double mse = 0.0;        // this step's loss, averaged over the draws
    double best_mse = 0.0;   // best seen so far, which is what is kept
    double elapsed_s = 0.0;

    // 10*log10(mse_start / best_mse): how far the *objective* has come.
    //
    // **Not an on-air figure and must not be shown as one.** Measured,
    // latent-domain MSE against a noiseless decode overstates recovered
    // picture quality by roughly 3x, and by a ratio that varies with
    // the image. It is here because it is free -- both terms are
    // already in hand -- and because a number that visibly climbs is
    // worth having while the operator waits. Present it as progress,
    // not as decibels earned.
    double objective_gain_db = 0.0;
};

// Called once per step. Returning false asks the loop to stop -- that
// is how a GUI's Cancel gets out without the optimizer knowing what a
// GUI is.
//
// It is also how a *deadline* that changes mid-run is expressed, which
// is what the app needs: optimization starts speculatively while the
// operator is still composing and runs to a generous budget, and when
// Send is clicked a shorter budget starts from the click. The loop
// deliberately does not model that -- `time_budget_s` is the outer
// backstop and the caller applies whichever bound is currently
// binding. Granularity is one step, so a post-Send budget shorter than
// that is not meaningful. See docs/latent-optimization.md.
using ProgressFn = std::function<bool(const Progress&)>;

enum class StopReason { Plateau, TimeBudget, MaxSteps, Cancelled };

std::string to_string(StopReason reason);

struct Options {
    double objective_snr_db = OBJECTIVE_SNR_DB;
    int channel_samples = CHANNEL_SAMPLES;
    double learning_rate = LEARNING_RATE;

    // Whichever fires first. **Not a fixed step count**: per-step cost
    // varies by an order of magnitude across the machines this ships
    // to, so a count that is seconds on a desktop is minutes on a small
    // board. The plateau test is the one that should normally fire; the
    // budget is what makes this safe inside a transmit workflow;
    // max_steps only backstops a loss that never plateaus.
    double time_budget_s = 20.0;
    int max_steps = 1000;
    int patience = 10;
    double min_rel_gain = 2e-3;

    std::uint64_t seed = 0;
};

struct Result {
    std::vector<double> latents;  // mode-length, unit RMS, ready for the modem
    int steps = 0;
    double seconds = 0.0;
    StopReason stop = StopReason::MaxSteps;
    double mse_start = 0.0;
    double mse_best = 0.0;

    // Improvement in the *objective*, which is not the on-air gain:
    // latent-domain MSE against a noiseless decode overstates what the
    // receiver sees by roughly 3x. Useful as progress, misleading as a
    // headline.
    double objective_gain_db() const;
};

// Better latents for this picture, starting from the encoder's.
//
// `latents` is at least this mode's length (the encoder's full output
// is fine; the tail is ignored). Returns the *best* iterate, not the
// last: the loss reported at a step belongs to the latents that went
// into it, so the final update is always unmeasured and returning it
// would sometimes ship a step past the minimum.
Result run(const GradFn& grad_fn, const std::vector<double>& latents,
           const config::ModeSpec& mode, const Options& opts = {},
           const ProgressFn& progress = nullptr);

}  // namespace sstvae::optimize

#endif
