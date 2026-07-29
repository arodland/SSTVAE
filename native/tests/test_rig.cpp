// Rig control's threading properties, against a fake radio.
//
// `docs/native-app.md`: *"tests/test_rig_controller.py's value is the
// scenario, not the transport: a rig that accepts and never answers."*
// So that is the scenario here, with a backend that blocks inside a
// call until the test lets it go -- no libhamlib, no serial port, no
// socket. The properties under test are the reference's: the caller
// never blocks on a wedged rig, teardown returns without waiting, a
// superseded worker cannot publish, and keying is not stuck behind
// status chatter.
//
// **Nothing here asserts on elapsed time.** The gated backend makes the
// interesting states reachable deterministically instead: if `stop()`
// ever grew a join, the wedged call would still be blocked when it ran
// and this test would hang rather than run slowly -- which the ctest
// TIMEOUT reports, and which is a far better signal than a threshold
// someone has to keep tuning.

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdio>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "check.hpp"
#include "rig/controller.hpp"

using namespace sstvae;

namespace {

// Shared between the test and the backend the controller owns, so the
// test can still see it after the controller has taken it.
struct Probe {
    std::mutex m;
    std::condition_variable cv;

    bool gate_open = false;      // release a blocked call
    bool in_call = false;        // a call is blocked right now
    int closes = 0;              // times close() has been called
    bool fail_open = false;
    bool fail_freq = false;
    double frequency = 14'230'000.0;
    std::vector<std::string> calls;  // in order

    void record(const std::string& what) {
        std::lock_guard<std::mutex> lock(m);
        calls.push_back(what);
    }

    std::vector<std::string> call_log() {
        std::lock_guard<std::mutex> lock(m);
        return calls;
    }

    void open_gate() {
        {
            std::lock_guard<std::mutex> lock(m);
            gate_open = true;
        }
        cv.notify_all();
    }

    void wait_until_blocked() {
        std::unique_lock<std::mutex> lock(m);
        cv.wait(lock, [&] { return in_call; });
    }

    int close_count() {
        std::lock_guard<std::mutex> lock(m);
        return closes;
    }
};

// A radio that accepts and then does not answer until told to.
class FakeRig final : public rig::RigBackend {
public:
    explicit FakeRig(std::shared_ptr<Probe> probe) : probe_(std::move(probe)) {}

    void open() override {
        probe_->record("open");
        if (probe_->fail_open) throw rig::RigError("no such device /dev/nope");
    }

    void close() noexcept override {
        std::lock_guard<std::mutex> lock(probe_->m);
        ++probe_->closes;
        probe_->calls.push_back("close");
        probe_->cv.notify_all();
    }

    void set_ptt(bool on) override {
        probe_->record(on ? "ptt-on" : "ptt-off");
        block();
    }

    double frequency_hz() override {
        probe_->record("freq");
        block();
        if (probe_->fail_freq) throw rig::RigError("rig timeout");
        return probe_->frequency;
    }

    std::string description() const override { return "Fake Rig"; }

private:
    // Blocks until the test opens the gate. A gate that is already open
    // makes the backend behave normally, which is what most of the
    // tests want.
    void block() {
        std::unique_lock<std::mutex> lock(probe_->m);
        if (probe_->gate_open) return;
        probe_->in_call = true;
        probe_->cv.notify_all();
        probe_->cv.wait(lock, [&] { return probe_->gate_open; });
        probe_->in_call = false;
    }

    std::shared_ptr<Probe> probe_;
};

std::unique_ptr<rig::RigBackend> make(std::shared_ptr<Probe> probe) {
    return std::make_unique<FakeRig>(std::move(probe));
}

// Collects what the controller publishes.
struct Published {
    std::mutex m;
    std::condition_variable cv;
    std::vector<std::optional<double>> frequencies;
    std::vector<std::string> statuses;

    rig::OnFrequency freq_fn() {
        return [this](std::optional<double> hz) {
            {
                std::lock_guard<std::mutex> lock(m);
                frequencies.push_back(hz);
            }
            cv.notify_all();
        };
    }
    rig::OnStatus status_fn() {
        return [this](const std::string& text) {
            {
                std::lock_guard<std::mutex> lock(m);
                statuses.push_back(text);
            }
            cv.notify_all();
        };
    }
    void wait_for_frequency() {
        std::unique_lock<std::mutex> lock(m);
        cv.wait(lock, [&] { return !frequencies.empty(); });
    }
    std::size_t frequency_count() {
        std::lock_guard<std::mutex> lock(m);
        return frequencies.size();
    }
    bool saw_status_containing(const std::string& needle) {
        std::lock_guard<std::mutex> lock(m);
        for (const std::string& s : statuses) {
            if (s.find(needle) != std::string::npos) return true;
        }
        return false;
    }
};

// --- backoff ----------------------------------------------------------------

void test_backoff_arithmetic() {
    // Straight from the reference: success returns to the configured
    // rate, failure doubles up to the cap. A rig that is not answering
    // rarely starts within one interval, and retrying at full rate turns
    // a misconfigured setup into a permanent stream of timeouts.
    check::equal(rig::next_interval(5.0, 5.0, true), 5.0, "rig/backoff: success resets");
    check::equal(rig::next_interval(40.0, 5.0, true), 5.0,
                 "rig/backoff: success resets from a long wait too");
    check::equal(rig::next_interval(5.0, 5.0, false), 10.0, "rig/backoff: failure doubles");
    check::equal(rig::next_interval(40.0, 5.0, false), rig::MAX_BACKOFF_S,
                 "rig/backoff: capped");
    check::equal(rig::next_interval(rig::MAX_BACKOFF_S, 5.0, false), rig::MAX_BACKOFF_S,
                 "rig/backoff: stays at the cap");
}

// --- the guarantee ----------------------------------------------------------

void test_stop_does_not_wait_for_a_wedged_rig() {
    // The whole reason this subsystem exists. If `stop()` joined, it
    // would still be inside the blocked call below and this test would
    // hang -- see the file header on why a hang is the right signal.
    auto probe = std::make_shared<Probe>();
    Published pub;
    {
        rig::RigController controller(pub.freq_fn(), pub.status_fn());
        controller.start(make(probe));
        probe->wait_until_blocked();

        controller.stop();
        check::is_true(!controller.running(),
                       "rig/stop: returns with the rig still mid-call");
        check::is_true(!controller.frequency_hz().has_value(),
                       "rig/stop: the cached frequency is dropped");

        // And a fresh session can start immediately, without waiting for
        // the abandoned worker -- applying new settings must not be
        // gated on the old rig giving up.
        auto probe2 = std::make_shared<Probe>();
        probe2->open_gate();
        controller.start(make(probe2));
        check::is_true(controller.running(), "rig/stop: a new session starts at once");
        controller.stop();
    }

    // Now let the abandoned worker unwind. Its exit path owns closing
    // the handle -- nobody is left to do it for it.
    probe->open_gate();
    {
        std::unique_lock<std::mutex> lock(probe->m);
        probe->cv.wait(lock, [&] { return probe->closes > 0; });
    }
    check::is_true(probe->close_count() >= 1,
                   "rig/stop: the abandoned worker closed its own handle");
}

void test_a_superseded_worker_does_not_publish() {
    // A reply that arrives after `stop()` must not overwrite the state
    // of whatever replaced it. The reference passes locals into the
    // thread to get this; here the stopped session simply refuses.
    auto probe = std::make_shared<Probe>();
    Published pub;
    {
        rig::RigController controller(pub.freq_fn(), pub.status_fn());
        controller.start(make(probe));
        probe->wait_until_blocked();
        controller.stop();

        probe->open_gate();  // the in-flight read now completes
        {
            std::unique_lock<std::mutex> lock(probe->m);
            probe->cv.wait(lock, [&] { return probe->closes > 0; });
        }
        check::equal(pub.frequency_count(), std::size_t{0},
                     "rig/stale: a stopped session publishes nothing");
    }
}

void test_keying_is_not_stuck_behind_status_chatter() {
    // PTT waits for at most the *one* operation already in flight, never
    // a queue of polls. With the rig wedged inside a frequency read,
    // keying is submitted and must be the very next thing the rig is
    // asked to do.
    auto probe = std::make_shared<Probe>();
    rig::RigController controller;
    rig::RigConfig config;
    config.poll_interval_s = 1.0;  // polls would pile up if they queued
    controller.start(make(probe), config);
    probe->wait_until_blocked();

    std::atomic<bool> keyed{false};
    std::atomic<bool> threw{false};
    std::thread keyer([&] {
        try {
            controller.set_ptt(true);
            keyed = true;
        } catch (const std::exception&) {
            threw = true;
        }
    });

    probe->open_gate();  // release the in-flight read; everything proceeds
    keyer.join();

    check::is_true(keyed.load() && !threw.load(), "rig/ptt: keying completed");
    const std::vector<std::string> calls = probe->call_log();
    // open, freq, ptt-on -- and nothing between the read and the key.
    std::size_t freq_idx = calls.size(), ptt_idx = calls.size();
    for (std::size_t i = 0; i < calls.size(); ++i) {
        if (calls[i] == "freq" && freq_idx == calls.size()) freq_idx = i;
        if (calls[i] == "ptt-on" && ptt_idx == calls.size()) ptt_idx = i;
    }
    check::is_true(ptt_idx < calls.size(), "rig/ptt: the rig was actually keyed");
    check::is_true(freq_idx < ptt_idx, "rig/ptt: the in-flight read came first");
    check::equal(ptt_idx - freq_idx, std::size_t{1},
                 "rig/ptt: keying waited for one operation, not a queue");
    controller.stop();
}

void test_keying_without_a_session_is_an_error_not_a_hang() {
    // TxEngine turns this into "PTT on failed", which is exactly right
    // for a transmission that cannot key the radio -- and much better
    // than blocking the transmit sequence on a rig that is not there.
    rig::RigController controller;
    bool threw = false;
    try {
        controller.set_ptt(true);
    } catch (const rig::RigError&) {
        threw = true;
    }
    check::is_true(threw, "rig/ptt: keying with no rig reports rather than waits");

    // And through the TxEngine-facing seam.
    const std::function<void(bool)> fn = controller.ptt_function();
    threw = false;
    try {
        fn(false);
    } catch (const rig::RigError&) {
        threw = true;
    }
    check::is_true(threw, "rig/ptt: the same through ptt_function()");
}

// --- ordinary operation -----------------------------------------------------

void test_polling_publishes_frequency_and_status() {
    auto probe = std::make_shared<Probe>();
    probe->open_gate();  // a responsive rig
    Published pub;
    rig::RigController controller(pub.freq_fn(), pub.status_fn());
    rig::RigConfig config;
    config.poll_interval_s = 1.0;
    controller.start(make(probe), config);

    pub.wait_for_frequency();
    check::is_true(controller.frequency_hz().has_value(),
                   "rig/poll: the cached frequency is available to read");
    check::is_true(pub.saw_status_containing("14.2300 MHz"),
                   "rig/poll: the status text carries the dial frequency");
    check::is_true(pub.saw_status_containing("Fake Rig"),
                   "rig/poll: and the rig announces itself on connecting");
    controller.stop();
}

void test_a_rig_that_cannot_be_opened_reports_and_stops() {
    auto probe = std::make_shared<Probe>();
    probe->open_gate();
    probe->fail_open = true;
    Published pub;
    rig::RigController controller(pub.freq_fn(), pub.status_fn());
    controller.start(make(probe));

    // Wait on the *status*, not on close(). A worker whose open() failed
    // has already returned, but the controller still holds a reference
    // to the session, so nothing is destroyed until stop() drops it --
    // waiting for close() here would wait forever. See the note on
    // `running()` in the header: a configured session is not the same
    // thing as a rig that is answering, and the status callback is what
    // reports health.
    {
        std::unique_lock<std::mutex> lock(pub.m);
        pub.cv.wait(lock, [&] {
            for (const std::string& s : pub.statuses) {
                if (s.find("/dev/nope") != std::string::npos) return true;
            }
            return false;
        });
    }
    check::equal(pub.frequency_count(), std::size_t{0},
                 "rig/open: nothing is published for a rig that never opened");
    controller.stop();
    check::is_true(pub.saw_status_containing("/dev/nope"),
                   "rig/open: the operator is told which device failed");
}

void test_polling_pauses_while_transmitting() {
    // Half duplex: the answer is not interesting mid-over, and some rigs
    // dislike CAT traffic while keyed. With polling paused, keying still
    // gets through -- which is the point of pausing.
    auto probe = std::make_shared<Probe>();
    probe->open_gate();
    Published pub;
    rig::RigController controller(pub.freq_fn(), pub.status_fn());
    rig::RigConfig config;
    config.poll_interval_s = 1.0;
    controller.start(make(probe), config);
    pub.wait_for_frequency();

    controller.pause_polling();
    controller.set_ptt(true);
    controller.set_ptt(false);
    controller.resume_polling();

    const std::vector<std::string> calls = probe->call_log();
    std::size_t on_idx = calls.size();
    for (std::size_t i = 0; i < calls.size(); ++i) {
        if (calls[i] == "ptt-on") on_idx = i;
    }
    check::is_true(on_idx < calls.size(), "rig/pause: keying works while paused");
    check::is_true(on_idx + 1 < calls.size() && calls[on_idx + 1] == "ptt-off",
                   "rig/pause: no poll slipped in between key and unkey");
    controller.stop();
}

void test_destruction_stops_a_running_controller() {
    auto probe = std::make_shared<Probe>();
    probe->open_gate();
    {
        rig::RigController controller;
        controller.start(make(probe));
    }
    std::unique_lock<std::mutex> lock(probe->m);
    probe->cv.wait(lock, [&] { return probe->closes > 0; });
    check::is_true(true, "rig/dtor: the backend is closed when the owner goes away");
}

}  // namespace

int main() {
    try {
        test_backoff_arithmetic();
        test_keying_without_a_session_is_an_error_not_a_hang();
        test_polling_publishes_frequency_and_status();
        test_a_rig_that_cannot_be_opened_reports_and_stops();
        test_stop_does_not_wait_for_a_wedged_rig();
        test_a_superseded_worker_does_not_publish();
        test_keying_is_not_stuck_behind_status_chatter();
        test_polling_pauses_while_transmitting();
        test_destruction_stops_a_running_controller();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return check::report("rig controller");
}
