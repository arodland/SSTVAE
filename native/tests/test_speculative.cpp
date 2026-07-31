// Speculative optimization: the coordination, not the arithmetic.
//
// `test_optimize.cpp` covers the loop. What is left here is the part
// that decides *which picture a result belongs to* -- and the failure
// it exists to prevent is transmitting the previous composition, which
// looks completely normal on screen and is only visible to whoever
// receives it.
//
// Nothing here waits on wall-clock durations to prove a property. The
// stub gradient blocks on a latch, so the test decides when a run is
// mid-flight and when it advances; the only deadlines present are the
// harness watchdog and a bounded wait helper, both of which exist to
// turn a hang into a message rather than a stalled CI job.

#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "check.hpp"
#include "config.hpp"
#include "images/types.hpp"
#include "latents/latents.hpp"
#include "optimize/speculative.hpp"

using namespace sstvae;

namespace {

const config::ModeSpec& mode_a() {
    for (const config::ModeSpec& m : config::MODES) {
        if (std::string(m.name) == "A") return m;
    }
    return config::MODES[0];
}

images::ImageArray picture(float fill) {
    images::ImageArray a;
    a.width = images::IMG_W;
    a.height = images::IMG_H;
    a.chw.assign(static_cast<std::size_t>(3) * images::IMG_H * images::IMG_W, fill);
    return a;
}

std::vector<double> latents_filled(const config::ModeSpec& m, double v) {
    return std::vector<double>(static_cast<std::size_t>(m.n_latents), v);
}

// Spin until `pred` or the bound expires. The bound is a watchdog, not
// an assertion: a passing run never reaches it, and a failing one gets
// a named check instead of a hung suite.
template <typename F>
bool wait_until(F pred, double bound_s = 20.0) {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::duration<double>(bound_s);
    while (std::chrono::steady_clock::now() < deadline) {
        if (pred()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return pred();
}

// A gradient stub the test drives: every call reports the target's
// fill value back through `seen_fill`, and optionally blocks until
// released, so a run can be held mid-flight while the picture changes.
struct Harness {
    std::mutex m;
    std::condition_variable cv;
    bool released = true;
    std::atomic<int> calls{0};
    std::atomic<float> seen_fill{-1.0f};

    optimize::GradFactory factory() {
        return [this](const images::ImageArray& target) {
            const float fill = target.chw.empty() ? -1.0f : target.chw[0];
            auto local = std::make_shared<int>(0);
            return [this, fill, local](const std::vector<float>& z,
                                       const std::vector<float>& w,
                                       std::vector<float>& grad, double& mse) {
                seen_fill = fill;
                ++calls;
                ++*local;
                {
                    std::unique_lock<std::mutex> lock(m);
                    cv.wait(lock, [this] { return released; });
                }
                grad.resize(z.size());
                for (std::size_t i = 0; i < z.size(); ++i) {
                    grad[i] = 0.02f * static_cast<float>(static_cast<int>(i % 5) - 2) *
                              w[i];
                }
                // The loss curve depends on the picture, which is what
                // lets a test tell "abandoned" from "finished on its
                // own". A dark picture improves geometrically forever
                // and so never plateaus; a light one flattens at once
                // and plateaus immediately.
                mse = (fill < 0.5f) ? std::pow(0.99, *local)
                                    : (*local <= 3 ? 1.0 / (1.0 + *local) : 0.25);
            };
        };
    }

    void hold() {
        std::lock_guard<std::mutex> g(m);
        released = false;
    }
    void release() {
        {
            std::lock_guard<std::mutex> g(m);
            released = true;
        }
        cv.notify_all();
    }
};

optimize::SpeculativeConfig fast_config() {
    optimize::SpeculativeConfig cfg;
    cfg.debounce_s = 0.0;      // the debounce is not what is under test here
    cfg.idle_budget_s = 60.0;
    cfg.send_budget_s = 60.0;
    cfg.options.channel_samples = 1;
    cfg.options.max_steps = 100000;
    cfg.options.patience = 100000;
    cfg.options.objective_snr_db = 40.0;
    return cfg;
}

void test_a_result_is_produced_and_is_mode_length() {
    check::current_step = "produces a result";
    Harness h;
    optimize::SpeculativeConfig cfg = fast_config();
    cfg.options.max_steps = 3;
    optimize::Speculative spec(h.factory(), cfg);

    spec.picture_changed(picture(0.25f), []{ return latents_filled(mode_a(), 0.3); }, mode_a());
    check::is_true(wait_until([&] { return spec.ready(); }), "a result arrives");

    const std::vector<double> got = spec.take_result();
    check::equal(got.size(), static_cast<std::size_t>(mode_a().n_latents),
                 "result is mode-length");
}

void test_an_edit_mid_run_invalidates_the_result() {
    check::current_step = "staleness";
    // The failure this class exists to prevent. Picture A is held
    // mid-run, picture B replaces it, and the abandoned run must
    // publish nothing -- the result the app collects has to be B's.
    Harness h;
    optimize::SpeculativeConfig cfg = fast_config();
    // The worker runs one job at a time, so a superseded run that
    // keeps going never lets its replacement start. The first picture
    // is the one whose loss improves forever (see Harness), so it can
    // only end by noticing it is stale -- a run that finishes on its
    // own would hide exactly the bug this is for.
    cfg.options.max_steps = 1000000;
    cfg.options.patience = 3;
    optimize::Speculative spec(h.factory(), cfg);

    h.hold();
    spec.picture_changed(picture(0.25f), []{ return latents_filled(mode_a(), 0.3); }, mode_a());
    check::is_true(wait_until([&] { return h.calls.load() > 0; }),
                   "the first run reached the gradient");
    check::close(std::vector<double>{h.seen_fill.load()}, std::vector<double>{0.25},
                 1e-6, "and it was working on the first picture");

    // Supersede it while it is blocked inside the gradient call.
    spec.picture_changed(picture(0.75f), []{ return latents_filled(mode_a(), 0.4); }, mode_a());
    check::is_true(spec.take_result().empty(),
                   "no result is offered for a superseded generation");
    h.release();

    check::is_true(wait_until([&] {
        return spec.ready() && !spec.take_result().empty();
    }), "the replacement run finishes");
    check::close(std::vector<double>{h.seen_fill.load()}, std::vector<double>{0.75},
                 1e-6, "the result belongs to the second picture");
}


void test_a_finished_result_is_withdrawn_by_a_later_edit() {
    check::current_step = "finished then edited";
    // The other half of the staleness rule, and the one that needs a
    // *completed* result to exercise: A finishes, then the operator
    // edits. A's latents must not be offered for B's generation even
    // though they are sitting right there.
    Harness h;
    optimize::SpeculativeConfig cfg = fast_config();
    cfg.options.max_steps = 3;
    optimize::Speculative spec(h.factory(), cfg);

    spec.picture_changed(picture(0.25f), []{ return latents_filled(mode_a(), 0.3); }, mode_a());
    check::is_true(wait_until([&] { return spec.ready(); }), "first run finished");
    check::is_true(!spec.take_result().empty(), "and produced latents");

    h.hold();  // keep the replacement from finishing and republishing
    spec.picture_changed(picture(0.75f), []{ return latents_filled(mode_a(), 0.4); }, mode_a());
    check::is_true(spec.take_result().empty(),
                   "the finished result is withdrawn once the picture changes");
    check::is_true(!spec.ready(), "and the caller is told to wait");
    h.release();
}

void test_request_send_is_safe_before_anything_is_scheduled() {
    check::current_step = "send with nothing to do";
    // Send with no picture must not wedge the caller: `ready` has to
    // be true so the GUI proceeds to transmit rather than waiting for
    // a run that will never exist.
    Harness h;
    optimize::Speculative spec(h.factory(), fast_config());
    spec.request_send();
    check::is_true(spec.ready(), "ready with nothing scheduled");
    check::is_true(spec.take_result().empty(), "and offers no latents");
}

void test_send_shortens_a_run_in_flight() {
    check::current_step = "send shortens";
    // A generous idle budget and a zero post-send budget: the run must
    // stop at the next step rather than continuing to the idle bound.
    Harness h;
    optimize::SpeculativeConfig cfg = fast_config();
    cfg.idle_budget_s = 3600.0;
    cfg.send_budget_s = 0.0;
    optimize::Speculative spec(h.factory(), cfg);

    spec.picture_changed(picture(0.5f), []{ return latents_filled(mode_a(), 0.3); }, mode_a());
    check::is_true(wait_until([&] { return h.calls.load() > 0; }),
                   "the run started");

    spec.request_send();
    check::is_true(wait_until([&] { return spec.ready(); }),
                   "send ends it without waiting for the idle budget");
    check::is_true(!spec.take_result().empty(), "and leaves latents to send");
}

void test_clear_abandons_and_offers_nothing() {
    check::current_step = "clear";
    Harness h;
    optimize::SpeculativeConfig cfg = fast_config();
    // The worker runs one job at a time, so a superseded run that
    // keeps going never lets its replacement start. The first picture
    // is the one whose loss improves forever (see Harness), so it can
    // only end by noticing it is stale -- a run that finishes on its
    // own would hide exactly the bug this is for.
    cfg.options.max_steps = 1000000;
    cfg.options.patience = 3;
    optimize::Speculative spec(h.factory(), cfg);

    h.hold();
    spec.picture_changed(picture(0.25f), []{ return latents_filled(mode_a(), 0.3); }, mode_a());
    check::is_true(wait_until([&] { return h.calls.load() > 0; }), "running");
    spec.clear();
    h.release();

    check::is_true(spec.take_result().empty(), "nothing is offered after clear");
    check::is_true(wait_until([&] { return spec.status().idle; }),
                   "and it settles back to idle");
}

void test_a_broken_gradient_falls_back_to_the_encoder_latents() {
    check::current_step = "artifact failure";
    // A missing or broken artifact costs the improvement, not the
    // picture: the operator must still be able to transmit.
    auto exploding = [](const images::ImageArray&) -> optimize::GradFn {
        throw std::runtime_error("no artifact");
    };
    optimize::Speculative spec(exploding, fast_config());

    const std::vector<double> encoder = latents_filled(mode_a(), 0.42);
    spec.picture_changed(picture(0.5f), [&] { return encoder; }, mode_a());
    check::is_true(wait_until([&] { return spec.ready(); }), "it still finishes");

    const std::vector<double> got = spec.take_result();
    check::equal(got.size(), encoder.size(), "and hands back mode-length latents");
    check::close(got, encoder, 1e-12, "which are the encoder's own");
}

void test_progress_reports_a_climbing_objective_gain() {
    check::current_step = "progress";
    // The number the GUI shows while the operator waits. It is an
    // objective value, not an on-air figure -- what is checked here is
    // only that it is reported and moves the right way.
    Harness h;
    optimize::SpeculativeConfig cfg = fast_config();
    cfg.options.max_steps = 6;
    std::atomic<int> notifications{0};
    optimize::Speculative spec(h.factory(), cfg, [&] { ++notifications; });

    spec.picture_changed(picture(0.5f), []{ return latents_filled(mode_a(), 0.3); }, mode_a());
    check::is_true(wait_until([&] { return spec.ready(); }), "run finished");

    const optimize::SpeculativeStatus st = spec.status();
    check::is_true(notifications.load() > 0, "the GUI was told something changed");
    check::is_true(st.progress.step > 0, "a step was reported");
    check::is_true(st.progress.objective_gain_db > 0.0,
                   "and the objective gain climbed above zero");
}

}  // namespace

int main() {
    check::report_crashes_instead_of_prompting();
    check::Watchdog watchdog(120, "speculative");
    test_a_result_is_produced_and_is_mode_length();
    test_an_edit_mid_run_invalidates_the_result();
    test_a_finished_result_is_withdrawn_by_a_later_edit();
    test_request_send_is_safe_before_anything_is_scheduled();
    test_send_shortens_a_run_in_flight();
    test_clear_abandons_and_offers_nothing();
    test_a_broken_gradient_falls_back_to_the_encoder_latents();
    test_progress_reports_a_climbing_objective_gain();
    return check::report("speculative");
}
