// Transmit-time latent optimization: the loop's decisions.
//
// The gradient itself has an oracle -- `torch.autograd.grad`, checked
// at export time by `scripts/export_onnx.py` -- so nothing here tries
// to re-derive it. What has no oracle is everything around it: which
// iterate is returned, why the loop stopped, whether a truncated mode's
// untransmitted groups stay zero, and whether the result is still on
// the unit-RMS shell the modem requires. Those are the parts that can
// be wrong while still producing a plausible picture.
//
// Driven by a stub gradient, so it runs in every build including
// `--no-codec` -- no onnxruntime, no downloaded artifact. That is the
// whole reason `optimize::run` takes a `GradFn` rather than owning a
// session.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "check.hpp"
#include "config.hpp"
#include "latents/latents.hpp"
#include "optimize/optimize.hpp"

using namespace sstvae;

namespace {

std::vector<double> flat_latents(int n, double value) {
    return std::vector<double>(static_cast<std::size_t>(n), value);
}

// A quadratic bowl centred on `target`: gradient 2*(z - target), so the
// loss falls monotonically and predictably. Enough structure to drive
// Adam, no dependence on a neural network.
optimize::GradFn bowl(const std::vector<float>& target, int* calls = nullptr) {
    return [target, calls](const std::vector<float>& z, const std::vector<float>& w,
                           std::vector<float>& grad, double& mse) {
        if (calls != nullptr) ++*calls;
        grad.assign(z.size(), 0.0f);
        double acc = 0.0;
        for (std::size_t i = 0; i < z.size(); ++i) {
            const double d = static_cast<double>(z[i]) - target[i];
            // The real graph multiplies the gradient by the weights;
            // the stub must too, or the mask test below would pass for
            // the wrong reason.
            grad[i] = static_cast<float>(2.0 * d) * w[i];
            acc += d * d * w[i];
        }
        mse = acc / static_cast<double>(z.size());
    };
}

// A loss that never improves: the plateau detector's job.
optimize::GradFn flat_loss() {
    return [](const std::vector<float>& z, const std::vector<float>&,
              std::vector<float>& grad, double& mse) {
        grad.assign(z.size(), 0.0f);
        mse = 1.0;
    };
}

const config::ModeSpec& mode(const char* name) {
    for (const config::ModeSpec& m : config::MODES) {
        if (std::string(m.name) == name) return m;
    }
    check::fail("mode", name);
    return config::MODES[0];
}

double rms_of(const std::vector<double>& v, std::size_t n) {
    double acc = 0.0;
    for (std::size_t i = 0; i < n; ++i) acc += v[i] * v[i];
    return std::sqrt(acc / static_cast<double>(n));
}

void test_it_improves_and_returns_unit_rms_latents() {
    check::current_step = "improves";
    const config::ModeSpec& m = mode("C");
    std::vector<float> target(latents::N_LATENTS, 0.0f);
    for (std::size_t i = 0; i < target.size(); ++i) {
        target[i] = (i % 7 == 0) ? 1.5f : -0.5f;
    }

    optimize::Options opts;
    opts.max_steps = 60;
    opts.time_budget_s = 1e9;   // let max_steps govern
    opts.patience = 1000;
    opts.channel_samples = 1;
    opts.objective_snr_db = 40.0;  // near-noiseless, so the bowl dominates

    const optimize::Result r =
        optimize::run(bowl(target), flat_latents(m.n_latents, 0.3), m, opts);

    check::is_true(r.mse_best < r.mse_start, "loss improved");
    check::equal(r.latents.size(), static_cast<std::size_t>(m.n_latents),
                 "result is mode-length");
    // The on-air contract. `Modem::modulate` renormalizes anyway, but a
    // result that needed it would mean the projection was not running.
    check::close(std::vector<double>{rms_of(r.latents, r.latents.size())},
                 std::vector<double>{1.0}, 1e-6,
                 "result is unit RMS");
}

void test_a_truncated_mode_leaves_the_untransmitted_groups_alone() {
    check::current_step = "mode mask";
    const config::ModeSpec& a = mode("A");
    std::vector<float> target(latents::N_LATENTS, 2.0f);

    optimize::Options opts;
    opts.max_steps = 30;
    opts.time_budget_s = 1e9;
    opts.patience = 1000;
    opts.channel_samples = 1;
    opts.objective_snr_db = 40.0;

    const optimize::Result r =
        optimize::run(bowl(target), flat_latents(a.n_latents, 0.1), a, opts);

    check::equal(r.latents.size(), static_cast<std::size_t>(a.n_latents),
                 "mode A result is one group long");
    // Unit RMS over the *transmitted* groups only -- normalizing across
    // all 132 channels would be a different problem than the radio
    // poses, and would show up here as an RMS of 1/sqrt(3).
    check::close(std::vector<double>{rms_of(r.latents, r.latents.size())},
                 std::vector<double>{1.0}, 1e-6,
                 "mode A result is unit RMS over its own groups");
}

void test_it_returns_the_best_iterate_not_the_last() {
    check::current_step = "best iterate";
    // Loss falls for five steps and then rises steeply. With patience
    // exhausted the loop stops *after* the minimum, so returning the
    // last iterate returns a worse one.
    //
    // The latents themselves have to be compared, not just `mse_best`:
    // that field is tracked separately, so a version that reports the
    // right number while shipping the wrong vector passes any check on
    // the number alone. (Confirmed by mutation -- returning `z` instead
    // of `best_z` left every other assertion here green.)
    std::vector<std::vector<float>> seen;
    auto worsening = [&seen](const std::vector<float>& z, const std::vector<float>&,
                             std::vector<float>& grad, double& mse) {
        seen.push_back(z);
        // Deliberately *non-uniform*. A constant gradient moves every
        // coordinate by the same amount, and the unit-RMS projection
        // then scales it straight back -- so every iterate is
        // identical, best and last included, and this test cannot tell
        // them apart. That is how the first version of it passed the
        // mutation it was written to catch.
        grad.resize(z.size());
        for (std::size_t i = 0; i < z.size(); ++i) {
            grad[i] = 0.02f * static_cast<float>(static_cast<int>(i % 5) - 2);
        }
        const std::size_t step = seen.size();
        mse = (step <= 5) ? 1.0 / static_cast<double>(step)
                          : 0.2 * static_cast<double>(step);
    };

    optimize::Options opts;
    opts.max_steps = 40;
    opts.time_budget_s = 1e9;
    opts.patience = 3;
    opts.channel_samples = 1;
    // Noise below float resolution, so the vector the gradient function
    // was handed *is* the iterate and can be compared to the result.
    opts.objective_snr_db = 200.0;

    const config::ModeSpec& m = mode("B");
    const optimize::Result r =
        optimize::run(worsening, flat_latents(m.n_latents, 0.2), m, opts);

    check::equal(optimize::to_string(r.stop), std::string("plateau"),
                 "stopped on plateau");
    check::close(std::vector<double>{r.mse_best}, std::vector<double>{0.2}, 1e-12,
                 "kept the minimum, not the last loss");
    check::is_true(r.steps > 5, "ran past the minimum before stopping");

    check::is_true(seen.size() >= 6, "recorded the iterates");
    const std::vector<float>& best = seen[4];   // step 5, where mse = 0.2
    const std::vector<float>& last = seen.back();
    std::vector<double> got(r.latents.begin(), r.latents.end());
    std::vector<double> want(best.begin(), best.begin() + r.latents.size());
    check::close(got, want, 1e-6, "returned the best iterate's latents");

    // ...and that this is a real distinction: the last iterate differs,
    // so the check above cannot pass for both.
    double spread = 0.0;
    for (std::size_t i = 0; i < r.latents.size(); ++i) {
        spread = std::max(spread, std::abs(static_cast<double>(best[i]) - last[i]));
    }
    check::is_true(spread > 1e-4, "best and last iterates actually differ");
}

void test_each_stop_reason_fires() {
    check::current_step = "stop reasons";
    const config::ModeSpec& m = mode("A");
    const std::vector<double> z0 = flat_latents(m.n_latents, 0.4);

    {   // plateau
        optimize::Options o;
        o.max_steps = 500;
        o.time_budget_s = 1e9;
        o.patience = 4;
        o.channel_samples = 1;
        const optimize::Result r = optimize::run(flat_loss(), z0, m, o);
        check::equal(optimize::to_string(r.stop), std::string("plateau"),
                     "plateau fires");
        check::is_true(r.steps < 500, "and stops early");
    }
    {   // max steps -- reached only because patience cannot fire
        optimize::Options o;
        o.max_steps = 7;
        o.time_budget_s = 1e9;
        o.patience = 1000;
        o.channel_samples = 1;
        std::vector<float> target(latents::N_LATENTS, 1.0f);
        const optimize::Result r = optimize::run(bowl(target), z0, m, o);
        check::equal(optimize::to_string(r.stop), std::string("max steps"),
                     "max steps fires");
        check::equal(r.steps, 7, "and ran them all");
    }
    {   // cancellation, via the progress callback
        optimize::Options o;
        o.max_steps = 500;
        o.time_budget_s = 1e9;
        o.patience = 1000;
        o.channel_samples = 1;
        std::vector<float> target(latents::N_LATENTS, 1.0f);
        const optimize::Result r =
            optimize::run(bowl(target), z0, m, o,
                          [](const optimize::Progress& p) { return p.step < 3; });
        check::equal(optimize::to_string(r.stop), std::string("cancelled"),
                     "cancel fires");
        check::equal(r.steps, 3, "at the step that asked to stop");
    }
    // The time budget is deliberately not asserted on: it is a
    // wall-clock deadline, and a test that waits for one is a timing
    // test. `StopReason::TimeBudget` is reachable by inspection -- the
    // same comparison as the others, on the same clock the loop uses.
}

void test_the_reported_loss_is_a_mean_over_channel_draws() {
    check::current_step = "mc averaging";
    // The *loss* must be the mean of the draws, because that number is
    // reported to the caller and drives the plateau test. A sum would
    // scale it with `channel_samples` and silently change what
    // "improving by 0.2%" means.
    //
    // Note what is deliberately NOT asserted: that averaging the
    // *gradient* changes the trajectory. It does not -- Adam's update
    // is m/sqrt(v) and so is invariant to a constant scaling of the
    // gradient, which means summing the draws takes identical steps.
    // An earlier version of this test claimed otherwise and passed a
    // mutation that summed them.
    const config::ModeSpec& m = mode("A");
    const std::vector<double> z0 = flat_latents(m.n_latents, 0.2);

    optimize::Options o;
    o.max_steps = 3;
    o.time_budget_s = 1e9;
    o.patience = 1000;

    int calls_one = 0, calls_eight = 0;
    auto constant_loss = [](int* calls) {
        return [calls](const std::vector<float>& z, const std::vector<float>&,
                       std::vector<float>& grad, double& mse) {
            ++*calls;
            grad.assign(z.size(), 0.0f);
            mse = 7.0;  // every draw, so the mean is 7 and a sum is not
        };
    };

    o.channel_samples = 1;
    const optimize::Result a = optimize::run(constant_loss(&calls_one), z0, m, o);
    o.channel_samples = 8;
    const optimize::Result b = optimize::run(constant_loss(&calls_eight), z0, m, o);

    check::equal(calls_one, 3, "one draw per step");
    check::equal(calls_eight, 24, "eight draws per step");
    check::close(std::vector<double>{a.mse_best}, std::vector<double>{7.0}, 1e-12,
                 "one draw reports the loss as-is");
    check::close(std::vector<double>{b.mse_best}, std::vector<double>{7.0}, 1e-12,
                 "eight draws report the mean, not the sum");
}


// The fading objective's draw. What can be wrong here is silent: a
// profile that is sampled but whose weights never reach the graph
// optimizes the flat channel while looking like it does not, and a
// weight applied without the matching noise scale (or the reverse)
// models a channel that does not exist. Both are checked against the
// declared distribution rather than against a golden number, since the
// RNG stream is deliberately not shared with Python.
void test_the_fading_profile_reaches_the_gradient() {
    const config::ModeSpec& m = mode("C");
    const std::vector<double> z0 = flat_latents(m.n_latents, 0.3);

    // Record the weight every draw actually presents to the graph.
    std::vector<float> seen;
    auto recorder = [&seen](const std::vector<float>& z, const std::vector<float>& w,
                            std::vector<float>& grad, double& mse) {
        grad.assign(z.size(), 0.0f);
        seen.insert(seen.end(), w.begin(), w.end());
        mse = 1.0;
    };

    optimize::Options o;
    o.max_steps = 4;
    o.channel_samples = 2;
    o.time_budget_s = 1e9;
    o.patience = 1000;
    o.fading_profile = &optimize::MEASURED_FADING_PROFILE;
    o.objective_snr_db = optimize::FADING_OBJECTIVE_SNR_DB;
    optimize::run(recorder, z0, m, o);

    // Every weight seen must be one of the profile's, times the mode
    // mask -- so 0 (untransmitted, though mode C has none) or one of
    // the nine. A profile that was drawn but not applied would show a
    // solid 1.0 here.
    std::size_t matched = 0;
    for (float w : seen) {
        for (int b = 0; b < optimize::FADING_BINS; ++b) {
            if (std::abs(w - optimize::MEASURED_FADING_PROFILE.weight[b]) < 1e-6f) {
                ++matched;
                break;
            }
        }
    }
    check::equal(matched, seen.size(), "every weight drawn comes from the profile");

    // And the histogram must match the declared probabilities. With
    // ~1.3M draws the sampling error is far below the 0.02 tolerance,
    // so a transposed or truncated table fails loudly.
    std::vector<double> hist(optimize::FADING_BINS, 0.0);
    for (float w : seen) {
        for (int b = 0; b < optimize::FADING_BINS; ++b) {
            if (std::abs(w - optimize::MEASURED_FADING_PROFILE.weight[b]) < 1e-6f) {
                hist[b] += 1.0;
                break;
            }
        }
    }
    for (double& h : hist) h /= static_cast<double>(seen.size());
    check::close(hist,
                 std::vector<double>(optimize::MEASURED_FADING_PROFILE.prob.begin(),
                                     optimize::MEASURED_FADING_PROFILE.prob.end()),
                 0.02, "the drawn weights follow the profile's probabilities");
}

// The fading profile is **on by default** (2026-08-04). Two things are
// pinned here, and the first is what caught the switchover: a null
// profile still reproduces the old flat channel exactly, and the
// default is genuinely not that -- so flipping the default back, or
// wiring the profile in without it reaching the draw, fails loudly
// rather than silently optimizing the wrong channel.
void test_the_fading_profile_is_the_default_and_null_restores_the_flat_one() {
    const config::ModeSpec& m = mode("C");
    const std::vector<double> z0 = flat_latents(m.n_latents, 0.3);
    std::vector<float> target(latents::N_LATENTS, 0.0f);
    for (std::size_t i = 0; i < target.size(); ++i) {
        target[i] = (i % 7 == 0) ? 1.5f : -0.5f;
    }

    optimize::Options flat;
    flat.max_steps = 6;
    flat.channel_samples = 2;
    flat.time_budget_s = 1e9;
    flat.patience = 1000;
    flat.clean_probe_every = 0;
    // The two always move together -- a null profile at the fading
    // SNR is a channel nobody measured.
    flat.fading_profile = nullptr;
    flat.objective_snr_db = optimize::OBJECTIVE_SNR_DB;

    const optimize::Result a = optimize::run(bowl(target), z0, m, flat);
    const optimize::Result b = optimize::run(bowl(target), z0, m, flat);
    check::close(a.latents, b.latents, 0.0, "the flat channel is reproducible");

    optimize::Options def;
    def.max_steps = flat.max_steps;
    def.channel_samples = flat.channel_samples;
    def.time_budget_s = flat.time_budget_s;
    def.patience = flat.patience;
    def.clean_probe_every = 0;
    check::is_true(def.fading_profile == &optimize::MEASURED_FADING_PROFILE,
                   "the default profile is the measured one");
    check::close(std::vector<double>{def.objective_snr_db},
                 std::vector<double>{optimize::FADING_OBJECTIVE_SNR_DB}, 1e-12,
                 "the default SNR is the fading one, not the flat one");

    const optimize::Result c = optimize::run(bowl(target), z0, m, def);
    double diff = 0.0;
    for (std::size_t i = 0; i < c.latents.size(); ++i) {
        diff = std::max(diff, std::abs(c.latents[i] - a.latents[i]));
    }
    check::is_true(diff > 1e-6, "the default differs from the flat channel");
}

// The clean probe. Its value is that it is a real PSNR number rather
// than a proxy, so what must hold is that it equals the noise-free loss
// ratio exactly -- and that it costs the extra graph calls it claims,
// since "free" would mean it is not running.
void test_the_clean_probe_measures_the_noise_free_loss() {
    const config::ModeSpec& m = mode("C");
    const std::vector<double> z0 = flat_latents(m.n_latents, 0.3);
    // Non-uniform on purpose: a target that is a constant is radially
    // symmetric, so the unit-RMS projection undoes every Adam step
    // exactly and the loss cannot fall. That is a property of the stub,
    // not of the optimizer, and it silently makes an improvement test
    // vacuous -- this one caught it by failing.
    std::vector<float> target(latents::N_LATENTS, 0.0f);
    for (std::size_t i = 0; i < target.size(); ++i) {
        target[i] = (i % 7 == 0) ? 1.5f : -0.5f;
    }

    optimize::Options o;
    o.max_steps = 4;
    o.channel_samples = 2;
    o.time_budget_s = 1e9;
    o.patience = 1000;
    o.clean_probe_every = 2;

    int calls = 0;
    double last_clean = -1.0, last_est = -1.0;
    optimize::run(bowl(target, &calls), z0, m, o, [&](const optimize::Progress& p) {
        last_clean = p.clean_gain_db;
        last_est = p.estimated_gain_db;
        return true;
    });

    // 4 steps x 2 draws = 8, plus one baseline probe and probes at
    // steps 1, 2 and 4 = 4 more. A probe that never ran would give 8.
    check::equal(calls, 12, "the probe costs one extra gradient call each time");
    check::is_true(last_clean > 0.0, "the clean gain climbs as the bowl is descended");
    check::close(std::vector<double>{last_est},
                 std::vector<double>{last_clean * optimize::RETENTION}, 1e-12,
                 "the estimate is the clean gain times RETENTION");

    // Off by default: no probe, no extra calls, and the field stays 0.
    int plain_calls = 0;
    double plain_clean = -1.0;
    optimize::Options q = o;
    q.clean_probe_every = 0;
    optimize::run(bowl(target, &plain_calls), z0, m, q,
                  [&](const optimize::Progress& p) {
                      plain_clean = p.clean_gain_db;
                      return true;
                  });
    check::equal(plain_calls, 8, "no probing means no extra calls");
    check::close(std::vector<double>{plain_clean}, std::vector<double>{0.0}, 0.0,
                 "clean_gain_db stays zero when probing is off");
}

}  // namespace

int main() {
    check::report_crashes_instead_of_prompting();
    check::Watchdog watchdog(60, "optimize");
    test_it_improves_and_returns_unit_rms_latents();
    test_a_truncated_mode_leaves_the_untransmitted_groups_alone();
    test_it_returns_the_best_iterate_not_the_last();
    test_each_stop_reason_fires();
    test_the_reported_loss_is_a_mean_over_channel_draws();
    test_the_fading_profile_reaches_the_gradient();
    test_the_fading_profile_is_the_default_and_null_restores_the_flat_one();
    test_the_clean_probe_measures_the_noise_free_loss();
    return check::report("optimize");
}
