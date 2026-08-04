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

#include <array>
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

// The objective's channel above assumes every latent arrives at full
// confidence with one noise level. A real fade does not: the
// demodulator reports a per-latent confidence weight and the error that
// comes with it, and the interleaver's whole job is to scatter a burst
// so that what reaches the decoder is i.i.d. *per-latent confidence*
// rather than a contiguous outage.
//
// This is that joint distribution, measured off the real demodulator
// (`scripts/derive_fading_profile.py`, which regenerates it) and
// re-derived across 36 conditions -- 6 images x {mpp, mpd} x 3/6/12 dB
// -- to check that 27 numbers from one transmission are safe to freeze
// into two implementations. Preset barely matters (the top bin differs
// by 0.03 between mpp and mpd); SNR cancels because `rel_sigma` is
// normalized by the top bin.
//
// Two things a parametric fade got wrong and this does not. The
// receiver almost never reports near-zero confidence -- under 3.5% of
// latents below w = 0.2, across every condition -- so a Rayleigh gain
// or a hard erasure models a population that barely exists. And where
// it *does* report low confidence it is conservative rather than
// overconfident, because the equalizer's pilot floor clamps the 1/|h|
// amplification; the overconfidence is in the middle of the range, at
// about 1.2x.
//
// **Keep in step with `sstvae/latent_optim.py`, which is normative.**
// Exact agreement is neither expected nor possible -- the two draw from
// different RNGs, which is already the contract for the flat channel --
// but the *distribution* must match or the two implementations are
// optimizing different objectives.
inline constexpr int FADING_BINS = 9;

struct FadingProfile {
    std::array<double, FADING_BINS> prob;       // sums to 1
    std::array<float, FADING_BINS> weight;      // confidence the rx reports
    std::array<float, FADING_BINS> rel_sigma;   // error, relative to top bin
};

inline constexpr FadingProfile MEASURED_FADING_PROFILE = {
    {0.009, 0.025, 0.039, 0.053, 0.065, 0.110, 0.115, 0.111, 0.473},
    {0.05f, 0.15f, 0.25f, 0.35f, 0.45f, 0.575f, 0.725f, 0.875f, 0.975f},
    {10.73f, 6.99f, 4.44f, 3.25f, 2.59f, 2.03f, 1.64f, 1.38f, 1.00f},
};

// The fading profile wants its own SNR, and it is **not** 5 dB. It
// carries 5-7 dB more noise at the same nominal number, so the flat
// channel's constant lands somewhere much deeper on this scale. Swept
// over the full 10-image corpus: +0.200 dB at 5, **+0.237 at 7.5**,
// +0.163 at 10, and +/-0.000 at 2.5.
//
// **The asymmetry is inverted from `OBJECTIVE_SNR_DB`'s, so do not
// carry that rule across.** There, too low costs little. Here, too low
// is where it dies -- 2.5 dB scores nothing and -0.314 dB on its worst
// image. If this has to move, move it *up*.
inline constexpr double FADING_OBJECTIVE_SNR_DB = 7.5;

// Fraction of a *clean* decode gain that survives to the receiver.
//
// A clean PSNR gain predicts the delivered one at r = +0.988, so this
// one number is the whole estimator -- but it is a genuine handwave and
// the reason is structural: measured over 6 images x 7 channel cells,
// retention decomposes almost exactly into an image term plus a channel
// term (spreads 0.126 and 0.139, residual 0.022), and **the sender
// cannot know the channel term**. It ranges 0.39 at mpp/3 dB to 0.77 at
// AWGN/12 dB, with a full per-cell range of 0.22-0.90.
//
// 0.50 is mpp at 6 dB: a fading channel at moderate SNR, chosen so the
// estimate under-promises on a good path rather than over-promising on
// a bad one. Everything about it is monotone -- more clean gain, more
// delivered gain; better channel, more retained -- so it ranks
// correctly even where it is numerically off.
//
// **Show it as approximate.** It is a point estimate of a quantity with
// a factor-of-two spread, and the operator's actual channel picks which
// end. A bare "+2.1 dB" claims precision this does not have.
inline constexpr double RETENTION = 0.50;

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
    // **Not an on-air figure and must not be shown as one.** It
    // overstates recovered picture quality by roughly 3x, by a ratio
    // that varies with the image, and it is not comparable across
    // objectives at all -- on one picture the flat objective reported
    // 5.58 dB for a delivered 1.84 while the fading one reported 3.97
    // for a delivered 3.29, ranking the two backwards. It is here
    // because it is free and because a number that visibly climbs is
    // worth having while the operator waits. Present it as progress,
    // never as decibels earned; `clean_gain_db` is the honest one.
    double objective_gain_db = 0.0;

    // 10*log10(clean_mse_start / clean_mse_now): the improvement in a
    // **noise-free** decode of the current iterate, in dB of PSNR
    // against the source picture.
    //
    // This is a real picture-quality number, not a proxy. It costs one
    // extra `GradFn` call every `clean_probe_every` steps -- the graph's
    // own `mse` output with the noise switched off is image-domain PSNR,
    // agreeing with a full `codec::decode` to 0.0013 dB, so no decoder
    // session is needed and a send-only station never fetches one.
    //
    // Zero when probing is off, and it stays at the value of the last
    // probe between probes rather than interpolating.
    double clean_gain_db = 0.0;

    // `clean_gain_db * RETENTION`: what the *receiver* is expected to
    // see. Read the warning on RETENTION before showing this to anyone.
    double estimated_gain_db = 0.0;
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
    // **These two move together.** `objective_snr_db` is on a different
    // scale for each channel -- the profile carries 5-7 dB more noise
    // at the same nominal number -- so pairing the fading profile with
    // 5.0, or a null profile with 7.5, silently optimizes a channel
    // nobody measured. Adopted 2026-08-04, worth +0.237 dB of recovered
    // picture over the flat pair across the 10-image corpus at
    // identical compute.
    double objective_snr_db = FADING_OBJECTIVE_SNR_DB;
    int channel_samples = CHANNEL_SAMPLES;

    // Null falls back to the flat channel -- the pre-2026-08-04
    // objective, kept reachable because every measurement older than
    // that was taken against it. Setting it to null means setting
    // `objective_snr_db` to `OBJECTIVE_SNR_DB` as well.
    const FadingProfile* fading_profile = &MEASURED_FADING_PROFILE;
    double learning_rate = LEARNING_RATE;

    // Whichever fires first. **Not a fixed step count**: per-step cost
    // varies by an order of magnitude across the machines this ships
    // to, so a count that is seconds on a desktop is minutes on a small
    // board. The plateau test is the one that should normally fire; the
    // budget is what makes this safe inside a transmit workflow;
    // max_steps only backstops a loss that never plateaus.
    // Steps between noise-free probes for `Progress::clean_gain_db`,
    // which is what the GUI shows. 0 disables them. Probing only ever
    // happens when a `ProgressFn` was supplied, so a caller with no UI
    // pays nothing for this default; with one, it is ~5% of the step
    // cost at the default 4 draws.
    int clean_probe_every = 5;

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
