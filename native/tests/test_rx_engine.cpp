// The live reception state machine.
//
// This is the part of the receiver that has no oracle in the golden
// vectors: everything the modem does is a function of its input, while
// the decode loop's job is to *decide* -- is this a new transmission or
// one already handled, has a blind reception stopped growing, does a
// finished one stay finished while its audio is still in the buffer.
// Those decisions have historically been where the duplicate-picture and
// stopped-early bugs lived, and none of them are visible to a
// round-trip test.
//
// It runs without onnxruntime, which is the point of the `Decoder`
// seam: a stub decoder means the state machine is checkable in every
// build, including `--no-codec`. Nothing here asserts on how *long*
// anything takes -- `Progress::polls` exists so progress through the
// state machine can be waited on directly. The one place a deadline
// appears is the harness watchdog, whose only job is to turn a hang
// into a message rather than a stalled CI job.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include "check.hpp"
#include "config.hpp"
#include "framing/framing.hpp"
#include "modem/modem.hpp"
#include "rx/engine.hpp"

using namespace sstvae;

namespace {

// Deterministic pseudo-random unit-RMS latents; same generator as
// test_modem_roundtrip.cpp, and for the same reason -- nothing here is
// compared against Python, it just has to be reproducible.
std::vector<double> test_latents(int n, std::uint64_t seed) {
    std::vector<double> out(static_cast<std::size_t>(n));
    std::uint64_t s = seed;
    for (double& v : out) {
        double acc = 0.0;
        for (int i = 0; i < 4; ++i) {
            s += 0x9E3779B97F4A7C15ULL;
            std::uint64_t z = s;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
            z = z ^ (z >> 31);
            acc += static_cast<double>(z >> 11) / 9007199254740992.0 - 0.5;
        }
        v = acc * 1.7;
    }
    double ms = 0.0;
    for (double v : out) ms += v * v;
    const double rms = std::sqrt(ms / static_cast<double>(out.size()));
    for (double& v : out) v /= rms;
    return out;
}

const config::ModeSpec& mode_a() { return config::MODES[0]; }

std::vector<double> transmission(const std::string& callsign, std::uint64_t seed) {
    const modem::Modem m;
    return m.modulate(test_latents(mode_a().n_latents, seed), mode_a(), true, callsign);
}

// A marker the decoder writes into every picture it produces, so a
// reception can be shown to carry the decoder's output rather than
// something the loop invented.
constexpr std::uint8_t kDecoderMark = 0x5A;

// No preamble/header at all -- forces decode_loop past find_new_
// reception (which needs one) and into the blind path on every poll,
// mirroring `_frames_only` in tests/test_listen_state_machine.py.
std::vector<double> frames_only(const std::string& callsign, std::uint64_t seed) {
    const std::vector<double> full = transmission(callsign, seed);
    const auto skip = static_cast<std::size_t>(
        config::LEADIN_SAMPLES + config::PREAMBLE_SAMPLES + config::HEADER_SAMPLES);
    return std::vector<double>(full.begin() + static_cast<std::ptrdiff_t>(skip), full.end());
}

// Everything a running loop and the test share.
struct Harness {
    rx::RingBuffer ring;
    rx::SharedState state;
    rx::StopFlag stop;
    rx::RxConfig config;

    std::mutex m;
    std::condition_variable cv;
    std::vector<rx::Reception> received;
    int decodes = 0;

    explicit Harness(double seconds) : ring(seconds) {
        // The buffer is pre-filled before the loop starts, so the poll
        // interval only sets how often a static buffer is re-examined.
        config.poll_interval = 0.02;
        config.end_grace = 0.2;
        config.blind_search_seconds = 6.0;
    }

    rx::Decoder decoder() {
        return [this](std::span<const double>, std::span<const double> w) {
            {
                std::lock_guard<std::mutex> lock(m);
                ++decodes;
            }
            cv.notify_all();
            images::Picture p(2, 2);
            p.rgb[0] = kDecoderMark;
            // Second marker: how many latents actually arrived, so a
            // picture can be tied to the decode that produced it.
            p.rgb[1] = static_cast<std::uint8_t>(
                std::count_if(w.begin(), w.end(), [](double v) { return v != 0.0; }) & 0xFF);
            return p;
        };
    }

    // A sink that deliberately declines to save. A finished reception
    // must be recorded as *handled* either way -- if that bookkeeping
    // ever moved into the saving branch, a GUI with autosave off would
    // re-decode and re-report the same picture on every poll.
    rx::Sink sink() {
        return [this](const rx::Reception& r) -> std::optional<std::string> {
            {
                std::lock_guard<std::mutex> lock(m);
                received.push_back(r);
            }
            cv.notify_all();
            return std::nullopt;
        };
    }

    // Wait for a condition on the shared state.
    //
    // The deadline is a watchdog, not an assertion: it exists so a
    // broken loop names *which* wait it died on instead of going quiet.
    // The hard bound on the job is the TIMEOUT property in
    // tests/CMakeLists.txt -- several waits each expiring here would
    // otherwise outlast the CI run.
    //
    // 120 s is about 3x the slowest single step measured on CI under
    // ASan (the whole test is ~150 s there, ~67 s here). It was 180 s
    // and still expired, because that job built at -O0: the fix was to
    // stop doing that rather than to keep raising this, since a
    // deadline that is only *just* long enough is a latency assertion
    // wearing a disguise. If this fires, find out why -- do not raise
    // it again without a measurement.
    bool until(const std::function<bool()>& pred, const std::string& what) {
        std::unique_lock<std::mutex> lock(m);
        const bool ok = cv.wait_for(lock, std::chrono::seconds(120), pred);
        if (!ok) check::fail(what, "timed out waiting for the loop to get there");
        return ok;
    }

    // Poll counts come from the state, not the harness mutex, so nudge
    // the waiters whenever one might have changed.
    void poll_watcher() {
        while (!stop.is_set()) {
            cv.notify_all();
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        cv.notify_all();
    }

    std::uint64_t polls() { return state.get().polls; }
    std::size_t count() {
        std::lock_guard<std::mutex> lock(m);
        return received.size();
    }
};

void write_all(rx::RingBuffer& ring, const std::vector<double>& x) {
    ring.write(std::span<const double>(x.data(), x.size()));
}

// Deterministic noise, same generator shape as test_noise_produces_nothing
// below -- used here to stand in for a branch whose antenna never
// acquires anything.
std::vector<double> noise(std::size_t n, std::uint64_t seed, double scale = 0.05) {
    std::vector<double> out(n);
    std::uint64_t s = seed;
    for (double& v : out) {
        s += 0x9E3779B97F4A7C15ULL;
        std::uint64_t z = s;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z = z ^ (z >> 31);
        v = scale * (static_cast<double>(z >> 11) / 9007199254740992.0 - 0.5);
    }
    return out;
}

// Two-ring counterpart of Harness, for decode_loop_diversity. A sink
// that returns a fixed path (rather than declining, like Harness's)
// because the debug-image tests need something for
// save_debug_image_to_dir_sink-style callers to key off of.
struct DiversityHarness {
    rx::RingBuffer ring_a;
    rx::RingBuffer ring_b;
    rx::SharedState state;
    rx::StopFlag stop;
    rx::RxConfig config;

    std::mutex m;
    std::condition_variable cv;
    std::vector<rx::Reception> received;
    int debug_images = 0;

    explicit DiversityHarness(double seconds) : ring_a(seconds), ring_b(seconds) {
        config.poll_interval = 0.02;
        config.end_grace = 0.2;
    }

    std::array<rx::RingBuffer*, 2> rings() { return {&ring_a, &ring_b}; }

    rx::Decoder decoder() {
        return [](std::span<const double>, std::span<const double> w) {
            images::Picture p(2, 2);
            p.rgb[0] = kDecoderMark;
            p.rgb[1] = static_cast<std::uint8_t>(
                std::count_if(w.begin(), w.end(), [](double v) { return v != 0.0; }) & 0xFF);
            return p;
        };
    }

    rx::Sink sink() {
        return [this](const rx::Reception& r) -> std::optional<std::string> {
            {
                std::lock_guard<std::mutex> lock(m);
                received.push_back(r);
            }
            cv.notify_all();
            return std::string("fake_reception.png");
        };
    }

    rx::DebugImageSink debug_sink() {
        return [this](const images::Picture&,
                     const std::optional<std::string>& saved_path) -> std::optional<std::string> {
            if (!saved_path) return std::nullopt;
            {
                std::lock_guard<std::mutex> lock(m);
                ++debug_images;
            }
            cv.notify_all();
            return std::string("fake_reception_diversity.png");
        };
    }

    bool until(const std::function<bool()>& pred, const std::string& what) {
        std::unique_lock<std::mutex> lock(m);
        const bool ok = cv.wait_for(lock, std::chrono::seconds(120), pred);
        if (!ok) check::fail(what, "timed out waiting for the loop to get there");
        return ok;
    }

    std::size_t count() {
        std::lock_guard<std::mutex> lock(m);
        return received.size();
    }
};

void test_a_clean_transmission_is_received_once() {
    Harness h(40.0);
    h.config.once = true;
    write_all(h.ring, transmission("KC2G", 1));

    // `once` makes this synchronous: the loop returns when it is done.
    rx::decode_loop(h.ring, h.decoder(), h.state, h.config, h.stop, h.sink());

    check::equal(h.received.size(), std::size_t{1}, "rx/once: exactly one reception");
    if (h.received.empty()) return;
    const rx::Reception& r = h.received[0];
    check::is_true(r.mode_name.has_value() && *r.mode_name == "A",
                   "rx/once: the header's mode was recovered");
    check::equal(r.callsign, std::string("KC2G"), "rx/once: the beacon's callsign");
    check::is_true(r.frames_received.has_value() &&
                       *r.frames_received == mode_a().n_frames,
                   "rx/once: every frame arrived on a clean channel");
    check::is_true(r.n_frames_expected.has_value() &&
                       *r.n_frames_expected == mode_a().n_frames,
                   "rx/once: the expected frame count");
    check::is_true(!r.image.empty() && r.image.rgb[0] == kDecoderMark,
                   "rx/once: the picture came from the decoder");

    const rx::Progress p = h.state.get();
    check::is_true(p.status == rx::Status::Done, "rx/once: final status is done");
    check::is_true(!p.saved_path.has_value(),
                   "rx/once: a sink that declines to save reports no path");
    check::is_true(p.image != nullptr, "rx/once: the picture is published for display");
}

void test_a_finished_reception_is_not_rediscovered() {
    // The regression that matters most here. The ring buffer goes on
    // holding a finished transmission's audio for the whole buffer
    // length, so without the finished-starts bookkeeping every
    // subsequent poll re-finds it, re-decodes it and reports it again.
    Harness h(40.0);
    write_all(h.ring, transmission("W1AW", 2));

    std::thread loop([&] {
        rx::decode_loop(h.ring, h.decoder(), h.state, h.config, h.stop, h.sink());
    });
    std::thread watcher([&] { h.poll_watcher(); });

    h.until([&] { return h.received.size() >= 1; }, "rx/dup: the first reception");
    const std::uint64_t after = h.polls();
    // Two further polls, not five. The buffer is static, so every poll
    // after the reception is handled takes an identical path through the
    // loop -- the first one that declines to report it again is the
    // evidence, and the second only confirms the first was not a fluke
    // of the reset window. Five was five full demodulations of the whole
    // buffer for no extra information, which is most of what made this
    // test expensive under a sanitizer.
    h.until([&] { return h.polls() >= after + 2; }, "rx/dup: two more polls");
    h.stop.set();
    loop.join();
    watcher.join();

    check::equal(h.count(), std::size_t{1},
                 "rx/dup: still-buffered audio is not received a second time");
}

void test_two_transmissions_are_both_received() {
    // Back to back in one buffer. The second must not be hidden by the
    // first: `sync::acquire` returns a single global argmax, so an
    // already-handled transmission can outrank a later one, which is why
    // the loop searches the spans its finished receptions do not claim.
    std::vector<double> audio = transmission("KC2G", 3);
    const std::vector<double> second = transmission("W1AW", 4);
    audio.insert(audio.end(), second.begin(), second.end());

    Harness h(80.0);
    write_all(h.ring, audio);

    std::thread loop([&] {
        rx::decode_loop(h.ring, h.decoder(), h.state, h.config, h.stop, h.sink());
    });
    std::thread watcher([&] { h.poll_watcher(); });

    h.until([&] { return h.received.size() >= 2; }, "rx/two: both receptions");
    const std::uint64_t after = h.polls();
    // Two, for the reason given in the duplicate-reception test above.
    h.until([&] { return h.polls() >= after + 2; }, "rx/two: two more polls");
    h.stop.set();
    loop.join();
    watcher.join();

    check::equal(h.count(), std::size_t{2}, "rx/two: exactly two, no repeats");
    if (h.count() != 2) return;
    std::lock_guard<std::mutex> lock(h.m);
    const bool both = (h.received[0].callsign == "KC2G" && h.received[1].callsign == "W1AW") ||
                      (h.received[0].callsign == "W1AW" && h.received[1].callsign == "KC2G");
    check::is_true(both, "rx/two: two different transmissions, not one twice");
}

void test_diversity_combines_two_branches_into_one_reception() {
    DiversityHarness h(40.0);
    h.config.once = true;
    const std::vector<double> x = transmission("TEST", 6);
    write_all(h.ring_a, x);
    write_all(h.ring_b, x);

    rx::decode_loop_diversity(h.rings(), h.decoder(), h.state, h.config, h.stop, h.sink());

    check::equal(h.received.size(), std::size_t{1}, "rx/div: exactly one reception");
    if (h.received.empty()) return;
    const rx::Reception& r = h.received[0];
    check::is_true(r.frames_received.has_value() &&
                       *r.frames_received == mode_a().n_frames,
                   "rx/div: every frame arrived combining two clean branches");
    check::equal(r.callsign, std::string("TEST"), "rx/div: the beacon's callsign");
}

void test_diversity_falls_back_to_single_branch_when_the_other_is_dead() {
    DiversityHarness h(40.0);
    h.config.once = true;
    const std::vector<double> x = transmission("SOLO", 7);
    write_all(h.ring_a, x);
    write_all(h.ring_b, noise(x.size(), 99));

    rx::decode_loop_diversity(h.rings(), h.decoder(), h.state, h.config, h.stop, h.sink());

    check::equal(h.received.size(), std::size_t{1},
                 "rx/div-fallback: still received with only one branch locked");
    if (h.received.empty()) return;
    check::equal(h.received[0].callsign, std::string("SOLO"), "rx/div-fallback: callsign");
}

void test_diversity_two_transmissions_are_both_received() {
    std::vector<double> audio = transmission("KC2G", 8);
    const std::vector<double> second = transmission("W1AW", 9);
    audio.insert(audio.end(), second.begin(), second.end());

    DiversityHarness h(80.0);
    write_all(h.ring_a, audio);
    write_all(h.ring_b, audio);

    std::thread loop([&] {
        rx::decode_loop_diversity(h.rings(), h.decoder(), h.state, h.config, h.stop, h.sink());
    });

    h.until([&] { return h.received.size() >= 2; }, "rx/div-two: both receptions");
    h.stop.set();
    loop.join();

    check::equal(h.count(), std::size_t{2}, "rx/div-two: exactly two, no repeats");
    if (h.count() != 2) return;
    std::lock_guard<std::mutex> lock(h.m);
    const bool both = (h.received[0].callsign == "KC2G" && h.received[1].callsign == "W1AW") ||
                      (h.received[0].callsign == "W1AW" && h.received[1].callsign == "KC2G");
    check::is_true(both, "rx/div-two: two different transmissions, not one twice");
}

void test_diversity_debug_image_written_only_when_both_branches_lock() {
    // Both branches lock: the debug sink is invoked once.
    {
        DiversityHarness h(40.0);
        h.config.once = true;
        const std::vector<double> x = transmission("BOTH", 10);
        write_all(h.ring_a, x);
        write_all(h.ring_b, x);
        rx::DebugImageSink debug = h.debug_sink();
        rx::decode_loop_diversity(h.rings(), h.decoder(), h.state, h.config, h.stop, h.sink(),
                                  &debug);
        check::equal(h.debug_images, 1, "rx/div-debug: written when both branches contribute");
    }
    // Only one branch locks: nothing to compare, so the debug sink is
    // never called.
    {
        DiversityHarness h(40.0);
        h.config.once = true;
        const std::vector<double> x = transmission("ONE", 11);
        write_all(h.ring_a, x);
        write_all(h.ring_b, noise(x.size(), 55));
        rx::DebugImageSink debug = h.debug_sink();
        rx::decode_loop_diversity(h.rings(), h.decoder(), h.state, h.config, h.stop, h.sink(),
                                  &debug);
        check::equal(h.debug_images, 0, "rx/div-debug: skipped with only one branch locked");
    }
}

void test_noise_produces_nothing() {
    // What is asserted here is that noise never *finishes* a reception,
    // and deliberately not that it never starts one.
    //
    // A preamble-shaped peak in noise clears the detection threshold
    // every few seed-minutes, and the 24-bit Golay-coded header behind
    // it decodes to a plausible mode occasionally rather than never. The
    // reference behaves the same way -- measured, 0 spurious header
    // locks in 4 vetted peaks over 12 seeds for Python and 0 in 4 for
    // this implementation, but one of the first seeds tried here landed
    // on a lock. So "the decoder was never invoked" and "the status
    // never left listening" are both true *most* of the time, which
    // makes them exactly the kind of assertion that turns into a
    // mystery failure months later.
    //
    // Nothing being reported as a finished reception is the property
    // that actually matters, and it is not probabilistic: a spurious
    // lock reports a handful of the mode's frames and then stops
    // advancing, so the header path never reaches its frame count and
    // the blind path has no beacon to place it.
    for (std::uint64_t seed : {1ULL, 12345ULL, 777ULL}) {
        Harness h(10.0);
        std::vector<double> noise(static_cast<std::size_t>(8.0 * config::FS));
        std::uint64_t s = seed;
        for (double& v : noise) {
            s += 0x9E3779B97F4A7C15ULL;
            std::uint64_t z = s;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
            z = z ^ (z >> 31);
            v = 0.2 * (static_cast<double>(z >> 11) / 9007199254740992.0 - 0.5);
        }
        write_all(h.ring, noise);

        std::thread loop([&] {
            rx::decode_loop(h.ring, h.decoder(), h.state, h.config, h.stop, h.sink());
        });
        std::thread watcher([&] { h.poll_watcher(); });

        // More polls than `end_grace` spans, so a blind reception that
        // had locked would have had time to be called finished.
        h.until([&] { return h.polls() >= 4; }, "rx/noise: four polls");
        h.stop.set();
        loop.join();
        watcher.join();

        check::equal(h.count(), std::size_t{0}, "rx/noise: nothing is ever received");
    }
}

void test_low_cpu_loop_receives() {
    // The cheap variant. The whole transmission is already buffered, so
    // its "wait until the transmission should have arrived" phase
    // completes immediately and `once` makes this synchronous too.
    Harness h(40.0);
    h.config.once = true;
    write_all(h.ring, transmission("VK3ABC", 5));

    rx::decode_loop_low_cpu(h.ring, h.decoder(), h.state, h.config, h.stop, h.sink());

    check::equal(h.received.size(), std::size_t{1}, "rx/lowcpu: one reception");
    if (h.received.empty()) return;
    const rx::Reception& r = h.received[0];
    check::is_true(r.mode_name.has_value() && *r.mode_name == "A", "rx/lowcpu: mode A");
    check::equal(r.callsign, std::string("VK3ABC"), "rx/lowcpu: callsign");
    check::is_true(r.frames_received.has_value() &&
                       *r.frames_received == mode_a().n_frames,
                   "rx/lowcpu: every frame");
    check::is_true(!r.image.empty() && r.image.rgb[0] == kDecoderMark,
                   "rx/lowcpu: the decoder's picture");
}

// A transmission whose tail never arrived: everything up to `keep_s`
// seconds of it, and no more. What a reception looks like when the
// operator unkeys early, when the band takes the signal away, or when
// the recording simply stops.
std::vector<double> truncated(const std::string& callsign, std::uint64_t seed,
                              double keep_s) {
    std::vector<double> full = transmission(callsign, seed);
    const auto keep = static_cast<std::size_t>(keep_s * config::FS);
    if (keep < full.size()) full.resize(keep);
    return full;
}

void test_a_reception_whose_decodes_stop_is_still_delivered() {
    // The watchdog regression. A partial reception is decoded, and then
    // its decodes stop -- here because its audio ages out of the ring
    // buffer, which is one of the two ways it happens in the field (the
    // other is a blind lock decaying under BLIND_SCORE_THRESHOLD once
    // the transmission is over). Nothing else can finish it: it never
    // reports all its frames, and with only a few seconds of it in a
    // buffer that never reaches the end of a mode-A transmission, the
    // sample-position deadline cannot fire either.
    //
    // Every completion test used to be evaluated inside the branch that
    // ran only when the current poll had produced a decode, so from the
    // moment the decodes stopped, "is this finished?" was never asked
    // again: the loop sat in Receiving indefinitely and the picture it
    // had already decoded was never handed to the sink. A hang and
    // autosave never firing are that one bug from two ends.
    Harness h(12.0);
    write_all(h.ring, truncated("KC2G", 61, 3.0));

    std::thread loop([&] {
        rx::decode_loop(h.ring, h.decoder(), h.state, h.config, h.stop, h.sink());
    });
    std::thread watcher([&] { h.poll_watcher(); });

    // Let it lock and decode the fragment at least once, so what
    // follows is a reception being abandoned rather than never found.
    // Waited on the *status*, not on the decode count: the loop decodes
    // before it publishes, so a decoder-side counter can be observed one
    // step ahead of the state it is standing in for.
    h.until([&] { return h.state.get().status == rx::Status::Receiving; },
            "rx/watchdog: the fragment puts the loop in receiving");
    check::equal(h.count(), std::size_t{0},
                 "rx/watchdog: a fragment is not a finished reception yet");

    // Past the ring buffer's whole length, so nothing of the
    // transmission is left to decode.
    write_all(h.ring, std::vector<double>(static_cast<std::size_t>(15.0 * config::FS)));

    h.until([&] { return h.received.size() >= 1; },
            "rx/watchdog: a reception whose decodes stopped is still delivered -- "
            "otherwise it is still sitting in receiving with its picture "
            "undelivered, which is both the hang and the autosave-never-fires "
            "report");
    h.stop.set();
    loop.join();
    watcher.join();

    check::equal(h.count(), std::size_t{1}, "rx/watchdog: delivered exactly once");
    if (h.count() != 1) return;
    const rx::Reception& r = h.received[0];
    check::is_true(r.mode_name.has_value() && *r.mode_name == "A",
                   "rx/watchdog: the header's mode survived into the delivery");
    check::is_true(r.frames_received.has_value() && *r.frames_received > 0 &&
                       *r.frames_received < mode_a().n_frames,
                   "rx/watchdog: delivered as the partial reception it is");
    check::is_true(!r.image.empty() && r.image.rgb[0] == kDecoderMark,
                   "rx/watchdog: the last good decode's picture, not an empty one");
    check::is_true(h.state.get().status == rx::Status::Listening,
                   "rx/watchdog: and the loop goes back to listening");
}

void test_low_cpu_does_not_wait_forever_for_audio_that_stops() {
    // decode_loop_low_cpu locks on the preamble and then stops doing any
    // DSP at all until the buffer holds the whole transmission. That is
    // a wait on something outside the loop's control -- if capture stops
    // (the device is unplugged, the stream dies, the host stops
    // delivering callbacks) the audio it is waiting for never arrives,
    // and unbounded it holds the receiver in Receiving forever with no
    // picture ever handed to the sink.
    //
    // A second of the transmission is missing, so the bound under test
    // is about a second plus end_grace rather than a whole mode's
    // duration -- this is not asserting on how long that takes, only
    // that it is finite. `once` makes the call synchronous: before the
    // fix it simply never returned.
    Harness h(60.0);
    h.config.once = true;
    const std::vector<double> full = transmission("W1AW", 62);
    const double short_by_s = 1.0;
    write_all(h.ring, truncated("W1AW", 62,
                                static_cast<double>(full.size()) / config::FS - short_by_s));

    rx::decode_loop_low_cpu(h.ring, h.decoder(), h.state, h.config, h.stop, h.sink());

    check::equal(h.received.size(), std::size_t{1},
                 "rx/lowcpu-watchdog: audio that stops arriving still ends the "
                 "reception rather than waiting for it forever");
    if (h.received.empty()) return;
    check::is_true(h.received[0].frames_received.has_value() &&
                       *h.received[0].frames_received < mode_a().n_frames,
                   "rx/lowcpu-watchdog: delivered as the partial reception it is");
}

void test_fresh_session_does_not_inherit_blind_evidence_from_a_prior_one() {
    // The app's "start receiving" button and ReceivePanel::resume_after_
    // transmit both work by discarding the whole RingBuffer and calling
    // decode_loop again from scratch (see rx_panel.cpp's start() /
    // resume_after_transmit(), and RingBuffer::clear()'s comment, which
    // records that clear() is *not* how that path works), rather than
    // clearing state in place. blind_acc is a plain local inside
    // decode_loop, so a brand-new call gets a clean one for free -- but
    // that is a property of scoping, not something pinned against a
    // future change that keeps decode_loop (and its locals) running
    // across a session boundary instead of restarting it, which is
    // exactly the scenario an indeterminate gap in real captured audio
    // (silence while transmitting, or whatever a fresh capture stream
    // first hands back) has no way to signal to a *stale* accumulator.
    //
    // Session 1 gets a real, complete mode-A transmission (no preamble/
    // header, to force the blind path) and is left to lock and finish,
    // so it isn't just idling. Session 2 is a brand-new RingBuffer +
    // SharedState + decode_loop call, exactly as start()/resume_after_
    // transmit() produce, fed nothing but noise: it must never report a
    // reception. If blind_acc's folded evidence ever leaked across that
    // boundary, session 2 would begin already most of the way to
    // session 1's lock rather than from nothing.
    Harness h1(40.0);
    h1.config.once = true;
    write_all(h1.ring, frames_only("KC2G", 41));

    rx::decode_loop(h1.ring, h1.decoder(), h1.state, h1.config, h1.stop, h1.sink());

    check::equal(h1.received.size(), std::size_t{1},
                 "rx/session-reset: session 1 should lock on and finish a real "
                 "transmission -- otherwise this isn't exercising real "
                 "blind-accumulator evidence");

    // A brand-new session: fresh RingBuffer, fresh SharedState, fresh
    // decode_loop call. Fed nothing but noise -- a fixed seed, so this
    // is deterministic rather than a rare-false-lock flake.
    Harness h2(10.0);
    std::vector<double> noise(static_cast<std::size_t>(8.0 * config::FS));
    std::uint64_t s = 4242;
    for (double& v : noise) {
        s += 0x9E3779B97F4A7C15ULL;
        std::uint64_t z = s;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z = z ^ (z >> 31);
        v = 0.2 * (static_cast<double>(z >> 11) / 9007199254740992.0 - 0.5);
    }
    write_all(h2.ring, noise);

    std::thread loop([&] {
        rx::decode_loop(h2.ring, h2.decoder(), h2.state, h2.config, h2.stop, h2.sink());
    });
    std::thread watcher([&] { h2.poll_watcher(); });

    h2.until([&] { return h2.polls() >= 4; }, "rx/session-reset: four polls on session 2");
    h2.stop.set();
    loop.join();
    watcher.join();

    check::equal(h2.count(), std::size_t{0},
                 "rx/session-reset: a fresh session fed only noise must not report "
                 "a reception -- blind_acc leaked evidence across a session "
                 "boundary");
}

void test_stop_interrupts_a_wait_in_progress() {
    // Shutting the receiver down must not cost a poll interval. If the
    // condition variable were replaced by a polled flag this would sit
    // here for an hour, which the suite reports as a hang -- that is the
    // failure signal, and it needs no assertion about elapsed time.
    rx::StopFlag stop;
    check::is_true(!stop.is_set(), "rx/stop: starts clear");

    std::thread setter([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        stop.set();
    });
    check::is_true(stop.wait(3600.0), "rx/stop: a wait is interrupted by set()");
    setter.join();

    check::is_true(stop.is_set(), "rx/stop: stays set");
    check::is_true(stop.wait(3600.0), "rx/stop: waiting on an already-set flag returns");
}

// The adaptive backoff, as arithmetic.
//
// Checked here rather than by running the loop with a slow decoder and
// timing it: that would assert on latency, and this is a decision, not
// a duration. The mutants worth catching are an inverted comparison
// (which would make a *fast* poll back off) and dropping the
// `poll_interval` floor (which would make a fast device poll faster
// than it ever did).
void test_poll_backoff() {
    sstvae::rx::RxConfig cfg;
    cfg.poll_interval = 5.0;

    // Default is off: the interval, whatever the last poll cost.
    check::is_true(std::abs(sstvae::rx::poll_wait(cfg, 0.0) - 5.0) < 1e-12,
                 "rx/backoff: default is the fixed interval");
    check::is_true(std::abs(sstvae::rx::poll_wait(cfg, 30.0) - 5.0) < 1e-12,
                 "rx/backoff: default ignores an expensive poll");

    cfg.max_decode_duty = 0.5;
    // A desktop-speed poll never reaches the cap.
    check::is_true(std::abs(sstvae::rx::poll_wait(cfg, 0.05) - 5.0) < 1e-12,
                 "rx/backoff: a cheap poll still waits the interval");
    // At the duty exactly, the interval is still what governs.
    check::is_true(std::abs(sstvae::rx::poll_wait(cfg, 5.0) - 5.0) < 1e-12,
                 "rx/backoff: a poll costing the interval is the boundary");
    // Past it, the wait matches the cost, so half the time is idle.
    check::is_true(std::abs(sstvae::rx::poll_wait(cfg, 9.0) - 9.0) < 1e-12,
                 "rx/backoff: a slow poll is followed by equal idle");

    cfg.max_decode_duty = 0.25;
    check::is_true(std::abs(sstvae::rx::poll_wait(cfg, 4.0) - 12.0) < 1e-12,
                 "rx/backoff: a quarter duty waits three times the cost");

    // An absurd request is floored rather than obeyed.
    cfg.max_decode_duty = 0.0;
    check::is_true(std::abs(sstvae::rx::poll_wait(cfg, 10.0) - 190.0) < 1e-12,
                 "rx/backoff: the duty floor bounds the wait");
}

// The progress bar is position, not fill: the last frame successfully
// received over the frames expected, never latents-received over
// latents-expected. Only the blind path can tell the two apart -- it
// decodes whatever frames it can place, with holes where the signal
// faded or where the transmission started before the buffer did -- and
// a fill fraction there is a completion percentage that is not one.
//
// Arithmetic, so it is checked as arithmetic (see `blind_progress`'s
// header comment), rather than inferred from a whole decode run.
void test_blind_progress_is_the_last_frame_reached() {
    const int total = config::MODES[config::N_MODES - 1].n_frames;
    const auto n_latents =
        static_cast<std::size_t>(config::MODES[config::N_MODES - 1].n_latents);

    auto weights_for = [&](const std::vector<int>& frames, double w) {
        std::vector<double> out(n_latents, 0.0);
        for (const int f : frames)
            for (const std::int64_t i : framing::slot_range_for_frame(f).indices)
                out[static_cast<std::size_t>(i)] = w;
        return out;
    };

    const rx::BlindProgress none = rx::blind_progress(std::vector<double>(n_latents, 0.0));
    check::is_true(none.metric == 0 && none.frac == 0.0,
                   "rx/progress: nothing received is zero progress");

    // At the threshold, not over it: a latent that only ties does not
    // count, the same cutoff the stall metric has always used.
    const rx::BlindProgress tied = rx::blind_progress(weights_for({0}, 0.5));
    check::is_true(tied.metric == 0 && tied.frac == 0.0,
                   "rx/progress: a weight at the threshold does not count");

    // Every other frame of mode B's range plus its last: half the
    // latents, but the transmission has been followed all the way to its
    // end. The old count-based fraction reported ~50% here.
    const int reach = config::MODES[1].n_frames;
    std::vector<int> sparse;
    for (int f = 0; f < reach; f += 2) sparse.push_back(f);
    sparse.push_back(reach - 1);
    const rx::BlindProgress half = rx::blind_progress(weights_for(sparse, 1.0));
    check::is_true(std::abs(half.frac - static_cast<double>(reach) / total) < 1e-12,
                   "rx/progress: half the latents, all the way to mode B's last frame");
    check::is_true(half.metric == static_cast<int>(sparse.size()) * config::LATENTS_PER_FRAME,
                   "rx/progress: the stall metric is still the confident count");

    // Tuned in at frame 400 of mode C and heard the rest: two thirds of
    // the latents are gone for good and the bar must still read full.
    std::vector<int> late;
    for (int f = 400; f < total; ++f) late.push_back(f);
    check::is_true(std::abs(rx::blind_progress(weights_for(late, 1.0)).frac - 1.0) < 1e-12,
                   "rx/progress: a late join reports where it is, not how much it has");

    // Retrospective decoding filling in frames *behind* the furthest one
    // is progress in quality but not in position; the bar must not move.
    const double reached = rx::blind_progress(weights_for({0, 300}, 1.0)).frac;
    check::is_true(std::abs(reached - 301.0 / total) < 1e-12,
                   "rx/progress: the furthest frame sets the bar");
    std::vector<int> filled;
    for (int f = 0; f <= 300; ++f) filled.push_back(f);
    check::is_true(
        std::abs(rx::blind_progress(weights_for(filled, 1.0)).frac - reached) < 1e-12,
        "rx/progress: backfill behind the furthest frame does not move it");
}

// `frame_of_latent` is the inverse of `slot_range_for_frame`, and the
// latents that never get a slot belong to no frame at all -- defaulting
// those to 0 rather than -1 would hand frame 0 thousands of latents
// that are never transmitted.
void test_frame_of_latent_inverts_slot_range_for_frame() {
    const std::span<const std::int32_t> table = framing::frame_of_latent();
    check::is_true(table.size() == static_cast<std::size_t>(
                                       config::MODES[config::N_MODES - 1].n_latents),
                   "framing/frame_of_latent: covers mode C's canonical range");
    for (const int f : {0, 1, 219, 220, 437, 659})
        for (const std::int64_t i : framing::slot_range_for_frame(f).indices)
            check::is_true(table[static_cast<std::size_t>(i)] == f,
                           "framing/frame_of_latent: inverts slot_range_for_frame");
    const auto dropped = std::count_if(table.begin(), table.end(),
                                       [](std::int32_t f) { return f < 0; });
    check::is_true(dropped == static_cast<std::ptrdiff_t>(config::LATENT_GROUPS) *
                                  config::DROPPED_LATENTS_PER_GROUP,
                   "framing/frame_of_latent: exactly the never-transmitted latents "
                   "belong to no frame");
}

}  // namespace

int main() {
    try {
        test_poll_backoff();
        test_frame_of_latent_inverts_slot_range_for_frame();
        test_blind_progress_is_the_last_frame_reached();
        test_stop_interrupts_a_wait_in_progress();
        test_a_clean_transmission_is_received_once();
        test_noise_produces_nothing();
        test_low_cpu_loop_receives();
        test_a_finished_reception_is_not_rediscovered();
        test_a_reception_whose_decodes_stop_is_still_delivered();
        test_low_cpu_does_not_wait_forever_for_audio_that_stops();
        test_two_transmissions_are_both_received();
        test_fresh_session_does_not_inherit_blind_evidence_from_a_prior_one();
        test_diversity_combines_two_branches_into_one_reception();
        test_diversity_falls_back_to_single_branch_when_the_other_is_dead();
        test_diversity_two_transmissions_are_both_received();
        test_diversity_debug_image_written_only_when_both_branches_lock();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return check::report("rx engine");
}
