// native/core/qrss/listen_api (the listener as a library, under
// `sstvae-qrss listen` and the app's QRSS window) without Python: stub
// receivers, an injected clock, real directories.
//
//   * push() from another thread, the clock advancing with the audio: the
//     tile comes, stop() says "audio ended" and state.json not listening,
//     within the bound and without abandoning anything; the three locks are
//     free after it; push() after stop() does nothing; stop() twice;
//   * stop() with a live receive held on a latch: within the bound, the
//     receive abandoned, the locks released while it is still held (a new
//     session takes the same directories at once), nothing of it reaching
//     the sink, state.json or the tiles once it is let go; push()'s latency
//     meanwhile (printed: the port's measurement);
//   * the same during a replay (the session's thread then waits for the
//     receive itself); in both, listener.log too ends with the stop's own
//     lines and stays so once the receive is let go;
//   * stop() with the session's thread held mid-step (not waiting for
//     audio): abandoned at the deadline, and nothing written when it goes on;
//   * interrupt(): "stopped", state.json not listening;
//   * a replay source that ends by itself: on_ended, "audio ended";
//   * start()'s refusals and notices in the script's texts and order: the
//     configuration, the tile directory held (and nothing else left taken),
//     the store held (no store, no passband), the passband held, a source
//     factory's refusal (every lock released), and the codec-less notice.
//
// tests/test_qrss_native_listen.py drives the CLI built on this side by side
// with qrss_listen.py.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <complex>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "check.hpp"
#include "qrss/qrss_config.hpp"
#include "qrss/dirlock.hpp"
#include "qrss/frame.hpp"
#include "qrss/listen_api.hpp"
#include "qrss/listener.hpp"
#include "qrss/live.hpp"
#include "qrss/pyjson.hpp"
#include "qrss/source.hpp"

using namespace sstvae;
namespace ls = sstvae::qrss::listen;
namespace lv = sstvae::qrss::live;
namespace fr = sstvae::qrss::frame;
namespace pj = sstvae::qrss::pyjson;
namespace types = sstvae::qrss::types;
namespace rcv = sstvae::qrss::receiver;
namespace aq = sstvae::qrss::acquire;
namespace dl = sstvae::qrss::dirlock;
namespace qcfg = sstvae::qrss::config;
namespace fs = std::filesystem;
using steady = std::chrono::steady_clock;

namespace {

fs::path g_tmp;

struct Step {
    explicit Step(const char* name) { check::current_step = name; }
};

double since(steady::time_point a) { return std::chrono::duration<double>(steady::now() - a).count(); }

// The text as Python's read_text gives it: the listener writes its text
// files in text mode, so on Windows "\r\n" reads back as "\n".
std::string read_text(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("cannot read " + p.string());
    std::string s(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>{});
#ifdef _WIN32
    std::string lf;
    lf.reserve(s.size());
    for (std::size_t i = 0; i < s.size(); ++i)
        if (!(s[i] == '\r' && i + 1 < s.size() && s[i + 1] == '\n')) lf += s[i];
    s = std::move(lf);
#endif
    return s;
}

fs::path fresh_dir(const std::string& name) {
    const fs::path d = g_tmp / name;
    std::error_code ec;
    fs::remove_all(d, ec);
    fs::create_directories(d);
    return d;
}

bool listening(const fs::path& state) {
    return pj::loads(read_text(state / "state.json")).at("listening").as_bool();
}

std::string last_log_line(const fs::path& state) {
    const std::string s = read_text(state / "listener.log");
    std::size_t end = s.size();
    while (end > 0 && s[end - 1] == '\n') --end;
    const std::size_t start = s.rfind('\n', end ? end - 1 : 0);
    return s.substr(start == std::string::npos ? 0 : start + 1, end - (start == std::string::npos ? 0 : start + 1));
}

// STORE/passband as the session spells it: joined with "/", as the C++'s
// pathlib joins on every system (Windows takes the mixed spelling).
std::string pb_text(const fs::path& root) { return (root / "store").string() + "/passband"; }

bool ends_with(const std::string& s, const std::string& tail) {
    return s.size() >= tail.size() && s.compare(s.size() - tail.size(), tail.size(), tail) == 0;
}

// Every lock of a listener on these directories is free: taken here, then let go.
bool lock_free(const fs::path& dir) {
    std::optional<dl::DirLock> l = dl::DirLock::try_acquire(dir.string());
    return l.has_value();
}

// --- the clock, the audio and the stubs -------------------------------------------------------

constexpr std::int64_t Q = 1954752;   // 2025-10-01T00:00Z
double t0() { return static_cast<double>(lv::slot_t0(Q)); }
double tiny_end() { return t0() + lv::frame_seconds(fr::get("tiny")) + lv::PB_TAIL_S + 0.5; }

struct Clock {
    std::shared_ptr<std::atomic<double>> t = std::make_shared<std::atomic<double>>(0.0);
    lv::Clock fn() const {
        auto p = t;
        return [p] { return p->load(); };
    }
};

std::vector<float> audio(std::int64_t n, std::int64_t seed) {
    std::vector<float> x(static_cast<std::size_t>(n));
    std::uint64_t h = 0x9E3779B97F4A7C15ULL * static_cast<std::uint64_t>(seed + 1);
    for (float& v : x) {
        h = h * 6364136223846793005ULL + 1442695040888963407ULL;
        v = static_cast<float>(1e-3 * (static_cast<double>(h >> 11) / 9007199254740992.0 - 0.5));
    }
    return x;
}

types::Detection det(double f) {
    types::Detection d;
    d.f_hz = f;
    return d;
}

types::PassResult pass(double f, std::int64_t n = 1024) {
    types::PassResult p;
    p.f_hz = f;
    p.uid = "u" + std::to_string(static_cast<long long>(f));
    p.report.snr2500_db = -12.0;
    p.report.z_ref = 20.0;
    p.w.assign(static_cast<std::size_t>(n), 1.0f);
    p.z.assign(static_cast<std::size_t>(n), 0.25f);
    return p;
}

lv::FinishResult finished_at(double f) {
    lv::FinishResult r;
    r.passes = {pass(f)};
    r.heard = 1.0;
    return r;
}

// Stub receivers that find a signal at 1500 Hz at once.
lv::Seams quick_stubs() {
    lv::Seams s;
    s.candidates = [](const rcv::Prepared&, auto, std::span<const aq::Hint>) {
        return std::vector<types::Detection>{det(1500.0)};
    };
    s.detect = [](const rcv::Prepared&, auto, const rcv::DetectOptions&) {
        return std::vector<types::Detection>{det(1500.0)};
    };
    s.receive_slot = [](const rcv::Prepared&, auto, const rcv::SlotOptions&) {
        return std::vector<types::PassResult>{pass(1500.0)};
    };
    s.finish = [](std::span<const std::complex<float>>, std::span<const std::uint8_t>, std::int64_t, std::int64_t,
                  const qrss::api::ReceiveConfig&, std::span<const types::Detection>, std::span<const aq::Hint>) {
        return finished_at(1500.0);
    };
    return s;
}

// A latch a stub receive waits on (60 s at most: a test that forgets to
// release it fails "timed out" rather than hanging).
struct Latch {
    std::mutex mu;
    std::condition_variable cv;
    bool entered = false, release = false, timed_out = false, returned = false;
    int calls = 0;

    void hold() {
        std::unique_lock lk(mu);
        ++calls;
        entered = true;
        cv.notify_all();
        if (!cv.wait_for(lk, std::chrono::seconds(60), [&] { return release; })) timed_out = true;
    }
    void done() {
        std::lock_guard lk(mu);
        returned = true;
        cv.notify_all();
    }
    bool is_entered() {
        std::lock_guard lk(mu);
        return entered;
    }
    bool wait_entered(double s) {
        std::unique_lock lk(mu);
        return cv.wait_for(lk, std::chrono::duration<double>(s), [&] { return entered; });
    }
    void let_go() {
        {
            std::lock_guard lk(mu);
            release = true;
        }
        cv.notify_all();
    }
    bool wait_returned(double s) {
        std::unique_lock lk(mu);
        return cv.wait_for(lk, std::chrono::duration<double>(s), [&] { return returned; });
    }
};

// What a session's sink got.
struct Lines {
    std::mutex mu;
    std::vector<std::pair<ls::LineKind, std::string>> got;
    std::atomic<bool> concurrent{false};
    std::atomic<int> inside{0};

    ls::Sink sink() {
        return [this](ls::LineKind kind, const std::string& text) {
            if (++inside > 1) concurrent = true;
            {
                std::lock_guard lk(mu);
                got.emplace_back(kind, text);
            }
            --inside;
        };
    }
    std::vector<std::pair<ls::LineKind, std::string>> all() {
        std::lock_guard lk(mu);
        return got;
    }
    std::size_t size() {
        std::lock_guard lk(mu);
        return got.size();
    }
    int count(ls::LineKind kind, const std::string& text) {
        int n = 0;
        for (const auto& [k, t] : all()) n += (k == kind && t == text);
        return n;
    }
    bool any_log_ending(const std::string& tail) {
        for (const auto& [k, t] : all())
            if (k == ls::LineKind::log && ends_with(t, tail)) return true;
        return false;
    }
};

ls::Options tiny_options(const fs::path& root, const Clock& clk) {
    ls::Options o;
    o.frame = "tiny";
    o.just_ce = true;
    o.refresh_s = 5.0;
    o.render = false;
    o.store_dir = (root / "store").string();
    o.state_dir = (root / "state").string();
    o.clock = clk.fn();
    o.seams = quick_stubs();
    return o;
}

std::optional<lv::Tile> tile_of(const ls::Session& s, const std::string& id) {
    for (const lv::Tile& t : s.tiles())
        if (t.id == id) return t;
    return std::nullopt;
}

// Wait (up to 30 s) until the session has taken everything pushed, so the
// clock, advanced with each push, never runs ahead of the audio the
// listener has: a loaded machine (or a sanitizer) otherwise lets the pusher
// stamp minutes the listener has not read, and its refreshes then come at
// times the test did not mean.
bool drained(const ls::Session& s) {
    const steady::time_point a = steady::now();
    while (s.backlog() != 0) {
        if (since(a) > 30.0) return false;
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
    return true;
}

// A replay's chunks of 0.25 s, stamped from `from` to `to`.
class ListSource final : public lv::Source {
public:
    ListSource(double from, double to, std::string name) : name_(std::move(name)) {
        const std::int64_t n = static_cast<std::int64_t>((to - from) * 4.0);
        for (std::int64_t k = 0; k < n; ++k) {
            const std::vector<float> f = audio(qcfg::FS / 4, 300 + k);
            chunks_.push_back(
                lv::Chunk{std::vector<double>(f.begin(), f.end()), from + 0.25 * static_cast<double>(k + 1)});
        }
    }
    std::optional<lv::Chunk> next(double) override {
        const std::size_t k = k_.load();
        if (k >= chunks_.size()) return std::nullopt;
        k_ = k + 1;
        return chunks_[k];
    }
    bool eof() const override { return k_.load() >= chunks_.size(); }
    std::string name() const override { return name_; }

private:
    std::string name_;
    std::vector<lv::Chunk> chunks_;
    std::atomic<std::size_t> k_{0};
};

// --- the tests ----------------------------------------------------------------------------------

// push() from another thread with the clock advancing; the tile; a clean stop.
void test_push_from_another_thread() {
    Step step("push: from another thread, then stop");
    const fs::path root = fresh_dir("push");
    Clock clk;
    const double start = t0() - 16.0;
    clk.t->store(start);
    Lines lines;
    std::atomic<int> ended_calls{0};
    ls::Options o = tiny_options(root, clk);
    o.on_ended = [&] { ++ended_calls; };
    std::unique_ptr<ls::Session> s = ls::Session::start(o, lines.sink());
    check::is_true(s->running() && !s->ended(), "push: running after start");
    check::equal(s->state_dir(), (root / "state").string(), "push: state_dir");
    check::is_true(s->store_root() == (root / "store").string(), "push: the store held");
    check::is_true(s->passband_dir() == pb_text(root), "push: the passband store held");
    check::is_true(!lock_free(root / "state") && !lock_free(root / "store") && !lock_free(root / "store" / "passband"),
                   "push: all three locks held while it runs");

    const double end = tiny_end() + 2.0;
    const std::int64_t n_chunks = static_cast<std::int64_t>((end - start) * 4.0);
    std::atomic<int> refused{0};
    std::atomic<bool> paced{true};
    std::thread pusher([&] {
        for (std::int64_t k = 0; k < n_chunks; ++k) {
            clk.t->store(start + 0.25 * static_cast<double>(k + 1));
            if (!s->push(audio(qcfg::FS / 4, k))) ++refused;
            if (!drained(*s)) paced = false;
        }
    });
    pusher.join();
    check::equal(refused.load(), 0, "push: every push taken");
    check::is_true(paced.load(), "push: the session keeps up with the pushes");
    // the end-of-slot receive (the finish worker) completes the tile
    std::optional<lv::Tile> t;
    const steady::time_point a = steady::now();
    while (since(a) < 30.0) {
        t = tile_of(*s, "1954752-1500");
        if (t && t->status == "complete") break;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    check::is_true(t && t->status == "complete", "push: the tile completes");
    check::equal(lines.count(ls::LineKind::notice, "tiles in " + (root / "state").string()), 1, "push: tiles in DIR");
    check::is_true(lines.any_log_ending(" listening: app capture (8 kHz), frame tiny, waveform CE (tiny)"),
                   "push: the listening line names the push source");
    check::is_true(lines.any_log_ending(" slot 2025-10-01T00:00:00Z: frame started"), "push: frame started");

    check::current_step = "push: stop";
    const steady::time_point b = steady::now();
    s->stop();
    const double took = since(b);
    std::printf("stop() after a clean run: %.3f s\n", took);
    check::is_true(took < ls::STOP_BOUND_S + 0.5, "push: stop() within the bound (" + std::to_string(took) + " s)");
    check::is_true(s->ended() && !s->running() && !s->abandoned() && !s->interrupted(), "push: ended cleanly");
    check::is_true(s->error() == nullptr, "push: no error");
    check::equal(ended_calls.load(), 1, "push: on_ended once");
    check::is_true(!listening(root / "state"), "push: state.json not listening");
    check::is_true(ends_with(last_log_line(root / "state"), " audio ended"), "push: audio ended, last in listener.log");
    check::is_true(lines.any_log_ending(" audio ended"), "push: audio ended reaches the sink");
    const std::vector<std::string> ll = s->lines();
    check::is_true(!ll.empty() && ends_with(ll.back(), " audio ended"), "push: lines() ends with it");
    check::is_true(lock_free(root / "state") && lock_free(root / "store") && lock_free(root / "store" / "passband"),
                   "push: every lock released");
    check::is_true(!lines.concurrent.load(), "push: the sink is called one line at a time");

    check::current_step = "push: after stop";
    check::is_true(!s->push(audio(100, 1)), "push: push() after stop() is refused");
    const steady::time_point c = steady::now();
    s->stop();
    s->interrupt();
    check::is_true(since(c) < 0.1, "push: a second stop() returns at once");
    check::equal(ended_calls.load(), 1, "push: on_ended still once");
    check::is_true(!s->interrupted(), "push: an interrupt after the end changes nothing");
    s.reset();
}

// stop() with a live receive held: bounded, abandoned, the locks released.
void test_stop_with_a_receive_held() {
    Step step("held: stop() with a receive held");
    const fs::path root = fresh_dir("held");
    Clock clk;
    const double start = t0() - 16.0;
    clk.t->store(start);
    Lines lines;
    auto latch = std::make_shared<Latch>();
    ls::Options o = tiny_options(root, clk);
    o.seams.receive_slot = [latch](const rcv::Prepared&, auto, const rcv::SlotOptions&) {
        latch->hold();
        latch->done();
        return std::vector<types::PassResult>{pass(1500.0)};
    };
    std::unique_ptr<ls::Session> s = ls::Session::start(o, lines.sink());
    std::int64_t k = 0;
    bool paced = true;
    while (!latch->is_entered() && k < 4 * 60) {   // (short of the frame's end, 69 s on)
        clk.t->store(start + 0.25 * static_cast<double>(k + 1));
        s->push(audio(qcfg::FS / 4, k));
        ++k;
        paced = drained(*s) && paced;
    }
    check::is_true(paced, "held: the session keeps up with the pushes");
    check::is_true(latch->wait_entered(30.0), "held: the receive started");

    // push() goes on while the receive is held (0.05 s chunks, short of the
    // slot's end: past it, the slot's finish would run beside the stop below
    // and, on a loaded machine, log after "audio ended")
    check::current_step = "held: pushing while the receive is held";
    std::vector<double> lat;
    double t = clk.t->load();
    for (int j = 0; j < 200; ++j) {
        t += 0.05;
        clk.t->store(t);
        const std::vector<float> x = audio(qcfg::FS / 20, 1000 + j);
        const steady::time_point a = steady::now();
        s->push(x);
        lat.push_back(since(a));
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    std::sort(lat.begin(), lat.end());
    std::printf("push() during a receive: p50 %.1f us, p95 %.1f us, max %.1f us (%zu pushes of 0.05 s)\n",
                lat[lat.size() / 2] * 1e6, lat[lat.size() * 95 / 100] * 1e6, lat.back() * 1e6, lat.size());
    check::is_true(lat.back() < 0.25, "held: push() never waits on the receive");
    check::is_true(clk.t->load() < tiny_end() - 1.0, "held: the clock is still short of the slot's end");

    // a second listener on the same tiles is refused; on the same store it runs without one
    check::current_step = "held: a second session";
    try {
        ls::Options o2 = tiny_options(root, clk);
        (void)ls::Session::start(o2, [](ls::LineKind, const std::string&) {});
        check::is_true(false, "held: a second session on the same --state is refused");
    } catch (const ls::Refusal& e) {
        check::equal(std::string(e.what()),
                     "another listener is writing tiles to " + (root / "state").string() +
                         "; stop it, or give this one its own --state DIR",
                     "held: the refusal's text");
    }
    {
        Lines l2;
        ls::Options o2 = tiny_options(root, clk);
        o2.state_dir = (root / "state2").string();
        std::unique_ptr<ls::Session> s2 = ls::Session::start(o2, l2.sink());
        check::equal(l2.count(ls::LineKind::notice, "another listener is using the store in " +
                                                        (root / "store").string() +
                                                        "; this one runs without a store (pass --store DIR for one "
                                                        "of its own)"),
                     1, "held: the store's notice");
        check::is_true(!s2->store_root() && !s2->passband_dir(), "held: no store, so no passband store");
        s2->stop();
        check::is_true(s2->ended() && !s2->abandoned(), "held: the second session stops cleanly");
    }
    const std::size_t lines_before = lines.size();

    check::current_step = "held: stop() with the receive held";
    const steady::time_point b = steady::now();
    s->stop();
    const double took = since(b);
    std::printf("stop() with a receive held: %.3f s (bound %.1f s)\n", took, ls::STOP_BOUND_S);
    check::is_true(took < ls::STOP_BOUND_S + 0.4, "held: stop() within the bound (" + std::to_string(took) + " s)");
    check::is_true(took > ls::STOP_BOUND_S - ls::SETTLE_S - 0.2, "held: it waited for the receive first");
    check::is_true(s->ended() && s->abandoned(), "held: ended, the receive abandoned");
    check::is_true(!latch->returned, "held: the receive is still held");
    check::is_true(lock_free(root / "state") && lock_free(root / "store") && lock_free(root / "store" / "passband"),
                   "held: every lock released with the receive still held");
    check::is_true(!listening(root / "state"), "held: state.json not listening (the session's thread was waiting)");
    {
        const std::string ll = last_log_line(root / "state");
        check::is_true(ends_with(ll, " audio ended"), "held: audio ended (the last line: " + ll + ")");
    }
    check::is_true(!s->push(audio(100, 2)), "held: push() after stop() is refused");

    // the same directories are free for a new session at once
    check::current_step = "held: a new session on the same directories";
    {
        Lines l3;
        std::unique_ptr<ls::Session> s3 = ls::Session::start(tiny_options(root, clk), l3.sink());
        check::is_true(s3->store_root().has_value() && s3->passband_dir().has_value(),
                       "held: the new one has both stores");
        const steady::time_point c = steady::now();
        s3->stop();
        check::is_true(since(c) < ls::STOP_BOUND_S + 0.4 && !s3->abandoned(), "held: and stops cleanly");
    }

    // let the abandoned receive go: nothing of it reaches the sink, state.json
    // or listener.log (which ends with a stop's own lines)
    check::current_step = "held: the abandoned receive returns";
    const std::string state_before = read_text(root / "state" / "state.json");
    const std::string log_before = read_text(root / "state" / "listener.log");
    const std::size_t sink_after_stop = lines.size();
    s.reset();   // (the session's threads outlive it)
    latch->let_go();
    check::is_true(latch->wait_returned(10.0), "held: the receive returns once let go");
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    check::is_true(!latch->timed_out, "held: released, not timed out");
    check::equal(lines.size(), sink_after_stop, "held: nothing more reached the sink");
    check::is_true(sink_after_stop >= lines_before, "held: (sanity)");
    check::equal(read_text(root / "state" / "state.json"), state_before, "held: state.json untouched after");
    check::is_true(!listening(root / "state"), "held: state.json still not listening");
    check::equal(read_text(root / "state" / "listener.log"), log_before, "held: listener.log untouched after");
    check::is_true(ends_with(last_log_line(root / "state"), " audio ended"), "held: listener.log still ends there");
    check::is_true(lock_free(root / "state") && lock_free(root / "store"), "held: the locks stay free");
}

// stop() while an end-of-slot receive runs that finishes within the bound:
// kept, and state.json (written after "audio ended") shows it.
void test_a_receive_that_finishes_within_the_bound() {
    Step step("finishes: a receive that ends inside the bound");
    const fs::path root = fresh_dir("finishes");
    Clock clk;
    const double start = t0() - 16.0;
    clk.t->store(start);
    Lines lines;
    auto entered = std::make_shared<std::atomic<bool>>(false);
    ls::Options o = tiny_options(root, clk);
    o.seams.receive_slot = [](const rcv::Prepared&, auto, const rcv::SlotOptions&) {
        return std::vector<types::PassResult>{};   // (no live tile: the end-of-slot receive makes it)
    };
    o.seams.finish = [entered](std::span<const std::complex<float>>, std::span<const std::uint8_t>, std::int64_t,
                               std::int64_t, const qrss::api::ReceiveConfig&, std::span<const types::Detection>,
                               std::span<const aq::Hint>) {
        entered->store(true);
        std::this_thread::sleep_for(std::chrono::milliseconds(400));
        return finished_at(1500.0);
    };
    std::unique_ptr<ls::Session> s = ls::Session::start(o, lines.sink());
    std::int64_t k = 0;
    bool paced = true;
    while (!entered->load() && k < 4 * 120) {
        clk.t->store(start + 0.25 * static_cast<double>(k + 1));
        s->push(audio(qcfg::FS / 4, k));
        ++k;
        paced = drained(*s) && paced;
    }
    check::is_true(paced && entered->load(), "finishes: the end-of-slot receive started");
    check::is_true(!tile_of(*s, "1954752-1500"), "finishes: no tile before it returns");
    const steady::time_point b = steady::now();
    s->stop();
    const double took = since(b);
    check::is_true(took < ls::STOP_BOUND_S + 0.4, "finishes: within the bound (" + std::to_string(took) + " s)");
    check::is_true(s->ended() && !s->abandoned(), "finishes: not abandoned");
    const pj::Value st = pj::loads(read_text(root / "state" / "state.json"));
    check::is_true(!st.at("listening").as_bool(), "finishes: state.json not listening");
    bool complete = false;
    for (const pj::Value& t : st.at("tiles").items())
        complete = complete || (t.at("id").as_string() == "1954752-1500" && t.at("status").as_string() == "complete");
    check::is_true(complete, "finishes: state.json shows the tile the receive completed after audio ended");
    check::is_true(lines.any_log_ending(" audio ended"), "finishes: audio ended");
    check::is_true(lock_free(root / "state") && lock_free(root / "store"), "finishes: locks released");
}

// The same during a replay, where the session's thread itself waits for the receive.
void test_stop_a_replay_with_a_receive_held() {
    Step step("replay held: stop() while a replay waits for its receive");
    const fs::path root = fresh_dir("replay_held");
    Clock clk;
    clk.t->store(1.79e9);
    auto latch = std::make_shared<Latch>();
    Lines lines;
    ls::Options o = tiny_options(root, clk);
    o.realtime = false;
    o.source = [](const ls::SourceContext&) {
        return std::make_unique<ListSource>(t0() - 16.0, tiny_end() + 2.0, "replay.wav");
    };
    o.seams.receive_slot = [latch](const rcv::Prepared&, auto, const rcv::SlotOptions&) {
        latch->hold();
        latch->done();
        return std::vector<types::PassResult>{pass(1500.0)};
    };
    std::unique_ptr<ls::Session> s = ls::Session::start(o, lines.sink());
    check::is_true(latch->wait_entered(30.0), "replay held: the receive started");
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const steady::time_point b = steady::now();
    s->stop();
    const double took = since(b);
    std::printf("stop() during a replay with a receive held: %.3f s\n", took);
    check::is_true(took < ls::STOP_BOUND_S + 0.4, "replay held: within the bound (" + std::to_string(took) + " s)");
    check::is_true(s->ended() && s->abandoned(), "replay held: abandoned");
    check::is_true(lock_free(root / "state") && lock_free(root / "store") && lock_free(root / "store" / "passband"),
                   "replay held: every lock released");
    check::is_true(!listening(root / "state"), "replay held: state.json not listening");
    check::is_true(ends_with(last_log_line(root / "state"), " audio ended"), "replay held: audio ended");
    const std::string state_before = read_text(root / "state" / "state.json");
    const std::string log_before = read_text(root / "state" / "listener.log");
    const std::size_t sink_after = lines.size();
    s.reset();
    latch->let_go();
    check::is_true(latch->wait_returned(10.0), "replay held: the receive returns once let go");
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    check::equal(lines.size(), sink_after, "replay held: nothing more reached the sink");
    check::equal(read_text(root / "state" / "state.json"), state_before, "replay held: state.json untouched after");
    check::equal(read_text(root / "state" / "listener.log"), log_before,
                 "replay held: listener.log untouched after (ending with the stop's audio ended)");
}

// A replay source whose name() -- which live::run asks for first, the
// session's thread holding its run lock -- waits on a latch: the stop's
// deadline finds that thread mid-step, neither waiting for audio nor on its
// last writes. The stop abandons it there, and when it goes on it writes
// nothing: no "listening" line, no state.json, though the locks are gone.
class HeldNameSource final : public lv::Source {
public:
    explicit HeldNameSource(std::shared_ptr<Latch> latch) : latch_(std::move(latch)), inner_(t0() - 16.0, t0() + 30.0, "replay.wav") {}
    std::optional<lv::Chunk> next(double timeout_s) override { return inner_.next(timeout_s); }
    bool eof() const override { return inner_.eof(); }
    std::string name() const override {
        if (!asked_.exchange(true)) {
            latch_->hold();
            latch_->done();
        }
        return inner_.name();
    }

private:
    std::shared_ptr<Latch> latch_;
    ListSource inner_;
    mutable std::atomic<bool> asked_{false};
};

void test_stop_with_the_thread_mid_step() {
    Step step("mid-step: stop() while the session's thread is mid-step");
    const fs::path root = fresh_dir("mid_step");
    Clock clk;
    clk.t->store(1.79e9);
    auto latch = std::make_shared<Latch>();
    Lines lines;
    ls::Options o = tiny_options(root, clk);
    o.realtime = false;
    o.source = [latch](const ls::SourceContext&) { return std::make_unique<HeldNameSource>(latch); };
    std::unique_ptr<ls::Session> s = ls::Session::start(o, lines.sink());
    check::is_true(latch->wait_entered(30.0), "mid-step: the session's thread is held");
    const steady::time_point b = steady::now();
    s->stop();
    const double took = since(b);
    check::is_true(took < ls::STOP_BOUND_S + 0.4, "mid-step: within the bound (" + std::to_string(took) + " s)");
    check::is_true(s->ended() && s->abandoned(), "mid-step: abandoned");
    check::is_true(!latch->returned, "mid-step: the thread is still held");
    check::is_true(lock_free(root / "state") && lock_free(root / "store") && lock_free(root / "store" / "passband"),
                   "mid-step: every lock released");
    const fs::path state = root / "state";
    check::is_true(!fs::exists(state / "listener.log") && !fs::exists(state / "state.json"),
                   "mid-step: nothing written yet");
    const std::size_t sink_after = lines.size();
    s.reset();   // (the session's thread outlives it)
    latch->let_go();
    check::is_true(latch->wait_returned(10.0), "mid-step: the thread goes on once let go");
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    check::is_true(!latch->timed_out, "mid-step: released, not timed out");
    check::is_true(!fs::exists(state / "listener.log"), "mid-step: no listener.log line once it goes on");
    check::is_true(!fs::exists(state / "state.json"), "mid-step: no state.json once it goes on");
    check::equal(lines.size(), sink_after, "mid-step: nothing more reached the sink");
}

// interrupt(): the script's KeyboardInterrupt.
void test_interrupt() {
    Step step("interrupt: stopped, not listening");
    const fs::path root = fresh_dir("interrupt");
    Clock clk;
    const double start = t0() - 16.0;
    clk.t->store(start);
    Lines lines;
    std::unique_ptr<ls::Session> s = ls::Session::start(tiny_options(root, clk), lines.sink());
    for (std::int64_t k = 0; k < 4 * 40; ++k) {
        clk.t->store(start + 0.25 * static_cast<double>(k + 1));
        s->push(audio(qcfg::FS / 4, k));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    const steady::time_point b = steady::now();
    s->interrupt();
    check::is_true(since(b) < ls::STOP_BOUND_S + 0.4, "interrupt: within the bound");
    check::is_true(s->ended() && s->interrupted() && !s->abandoned(), "interrupt: interrupted");
    check::is_true(!listening(root / "state"), "interrupt: state.json not listening");
    check::is_true(ends_with(last_log_line(root / "state"), " stopped"), "interrupt: stopped, last in listener.log");
    check::is_true(lines.any_log_ending(" stopped") && !lines.any_log_ending(" audio ended"),
                   "interrupt: stopped, and no audio ended");
    check::is_true(lock_free(root / "state") && lock_free(root / "store"), "interrupt: locks released");
}

// A replay that ends by itself.
void test_a_replay_ends_by_itself() {
    Step step("replay: ends by itself");
    const fs::path root = fresh_dir("replay");
    Clock clk;
    clk.t->store(1.79e9);
    Lines lines;
    std::atomic<int> ended_calls{0};
    std::atomic<bool> slept{false};
    ls::Options o = tiny_options(root, clk);
    o.realtime = false;
    o.source = [&slept](const ls::SourceContext& ctx) {
        // the context's sleep returns at once with nothing to wait for... and its clock is the session's
        ctx.sleep(0.0);
        slept = ctx.clock() == 1.79e9;
        return std::make_unique<ListSource>(t0() - 16.0, tiny_end() + 2.0, "replay.wav");
    };
    o.on_ended = [&] {
        // (slow on purpose: wait_ended() returns only after it has)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        ++ended_calls;
    };
    std::unique_ptr<ls::Session> s = ls::Session::start(o, lines.sink());
    check::is_true(slept.load(), "replay: the factory's context carries the session's clock");
    check::is_true(s->wait_ended(60.0), "replay: ends");
    check::equal(ended_calls.load(), 1, "replay: on_ended once");
    check::is_true(!s->abandoned() && !s->interrupted() && s->error() == nullptr, "replay: cleanly");
    const std::optional<lv::Tile> t = tile_of(*s, "1954752-1500");
    check::is_true(t && t->status == "complete", "replay: the tile complete");
    check::is_true(lines.any_log_ending(" listening: replay.wav, frame tiny, waveform CE (tiny)"),
                   "replay: listening names the source");
    check::is_true(ends_with(last_log_line(root / "state"), " audio ended"), "replay: audio ended");
    check::is_true(!listening(root / "state"), "replay: state.json not listening");
    check::is_true(!s->push(audio(10, 1)), "replay: push() is for the push source only");
    check::is_true(lock_free(root / "state") && lock_free(root / "store"), "replay: locks released");
}

// start()'s refusals and notices.
void test_refusals_and_notices() {
    Step step("start: refusals");
    const fs::path root = fresh_dir("refusals");
    Clock clk;
    clk.t->store(t0());
    auto refused = [&](ls::Options o, const std::string& want, const std::string& what) {
        try {
            (void)ls::Session::start(std::move(o), [](ls::LineKind, const std::string&) {});
            check::is_true(false, what + ": refused");
        } catch (const ls::Refusal& e) {
            check::equal(std::string(e.what()), want, what);
        }
    };
    ls::Options o = tiny_options(root, clk);
    o.frame = "x";
    refused(o, "--frame 'x': one of full, medium, short, tiny, l-full, l-medium, l-short, l-tiny", "bad --frame");
    o = tiny_options(root, clk);
    o.frame = "full-v1";
    o.just_ce = false;
    refused(o,
            "--frame 'full-v1' is CE only: waveform L has no frame of that shape (L frames: l-full, l-medium, "
            "l-short, l-tiny); add --just-ce to receive it",
            "CE-only frame without --just-ce");
    o.just_ce = true;
    o.just_l = true;
    refused(o, "argument --just-l: not allowed with argument --just-ce", "--just-ce with --just-l");
    o = tiny_options(root, clk);
    o.refresh_s = 0.0;
    refused(o, "--refresh must be positive", "--refresh 0");
    o.refresh_s = -1.0;
    refused(o, "--refresh must be positive", "--refresh -1");
    o = tiny_options(root, clk);
    o.precision = "fp64";
    refused(o, "argument --precision: invalid choice: 'fp64' (choose from 'fp32', 'fp16', 'int8')", "--precision");
    check::is_true(!fs::exists(root / "state") && !fs::exists(root / "store"),
                   "start: a configuration refusal takes nothing");

    // the configuration as the script builds it
    o = tiny_options(root, clk);
    o.frame = "l-short";
    o.just_ce = false;
    const lv::LiveConfig cfg = ls::build_config(o);
    check::equal(cfg.spec.name(), std::string("short"), "build_config: an L name names its CE twin");
    check::is_true(cfg.waveforms && cfg.waveforms->size() == 2, "build_config: both waveforms");
    check::is_true(cfg.background && cfg.finish_process && cfg.refresh_s == 5.0, "build_config: background, refresh");
    ls::Options d;
    d.store_dir = "/s/x/";
    check::equal(ls::store_root(d), std::string("/s/x"), "store_root: as Path() prints it");
    check::equal(ls::state_dir(d), std::string("/s/x/live"), "state_dir: STORE/live");
    d.state_dir = "";
    check::equal(ls::state_dir(d), std::string("."), "state_dir: Path('') is '.'");

    check::current_step = "start: the tile directory held";
    {
        std::optional<dl::DirLock> held = dl::DirLock::try_acquire((root / "state").string());
        refused(tiny_options(root, clk),
                "another listener is writing tiles to " + (root / "state").string() +
                    "; stop it, or give this one its own --state DIR",
                "the tile directory held");
        check::is_true(lock_free(root / "store"), "start: refused before the store's lock");
    }

    check::current_step = "start: the passband store held";
    {
        std::optional<dl::DirLock> held = dl::DirLock::try_acquire((root / "store" / "passband").string());
        Lines l;
        std::unique_ptr<ls::Session> s = ls::Session::start(tiny_options(root, clk), l.sink());
        const auto got = l.all();
        check::is_true(!got.empty() && got[0].first == ls::LineKind::notice &&
                           got[0].second == "another listener is writing the passband store in " +
                                                pb_text(root) + "; this one keeps none",
                       "start: the passband store's notice, first");
        check::is_true(s->store_root().has_value() && !s->passband_dir(), "start: a store, no passband store");
        s->stop();
    }

    check::current_step = "start: a source factory's refusal";
    {
        ls::Options f = tiny_options(root, clk);
        f.source = [](const ls::SourceContext&) -> std::unique_ptr<lv::Source> {
            throw ls::Refusal("--wav needs --start or --slot");
        };
        refused(f, "--wav needs --start or --slot", "the factory's refusal");
        check::is_true(lock_free(root / "state") && lock_free(root / "store") && lock_free(root / "store" / "passband"),
                       "start: every lock released after it");
    }

    check::current_step = "start: pictures asked for, no renderer";
    {
        ls::Options f = tiny_options(root, clk);
        f.render = true;
        Lines l;
        std::unique_ptr<ls::Session> s = ls::Session::start(f, l.sink());
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        check::equal(l.count(ls::LineKind::notice, ls::NO_PICTURES_NOTICE), 1, "start: the codec-less notice, once");
        s->stop();
        f.render = false;
        Lines l2;
        s = ls::Session::start(f, l2.sink());
        s->stop();
        check::equal(l2.count(ls::LineKind::notice, ls::NO_PICTURES_NOTICE), 0, "start: none with --no-pictures");
    }
}

// The PushSource alone: StreamSource's stamps, end(), and a bad timeout.
void test_push_source() {
    Step step("PushSource");
    Clock clk;
    clk.t->store(1000.0);
    ls::PushSource p("p", clk.fn(), 0.0);
    check::is_true(!p.next(0.0).has_value() && !p.eof(), "PushSource: nothing yet");
    check::is_true(p.push(std::vector<float>(800, 0.5f)), "PushSource: a push");
    check::is_true(p.push(std::span<const float>{}), "PushSource: an empty push is fine");
    clk.t->store(1000.1);
    check::is_true(p.push(std::vector<float>(800, 0.25f)), "PushSource: another");
    p.end();
    p.end();
    std::vector<lv::Chunk> got;
    while (std::optional<lv::Chunk> c = p.next(0.0)) got.push_back(std::move(*c));
    std::size_t n = 0;
    for (const lv::Chunk& c : got) n += c.x.size();
    check::equal(n, std::size_t{1600}, "PushSource: every sample, in order");
    check::is_true(!got.empty() && got.back().t_end <= 1000.1 + 1e-9, "PushSource: stamped by the clock");
    check::is_true(p.eof() && !p.push(std::vector<float>(10, 0.f)), "PushSource: ended");
    check::is_true(p.error() == nullptr, "PushSource: no error");
    bool threw = false;
    try {
        (void)p.next(-1.0);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check::is_true(threw, "PushSource: a negative timeout is refused");
    // a clock that throws ends the stream with its error
    ls::PushSource bad("b", [] () -> double { throw std::runtime_error("no clock"); }, 0.0);
    check::is_true(bad.push(std::vector<float>(10, 0.f)), "PushSource: taken");
    check::is_true(bad.error() != nullptr && bad.eof(), "PushSource: the clock's error ends it");
}

}  // namespace

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;
    check::report_crashes_instead_of_prompting();
    // ~10x the measured runtime: expiring means wedged, and names the step
    check::Watchdog watchdog(110.0, "qrss_listen_api");
    try {
        g_tmp = fs::temp_directory_path() / ("qrss_listen_api_test_" + std::to_string(std::random_device{}()));
        fs::create_directories(g_tmp);
        test_push_source();
        test_refusals_and_notices();
        test_push_from_another_thread();
        test_stop_with_a_receive_held();
        test_stop_a_replay_with_a_receive_held();
        test_stop_with_the_thread_mid_step();
        test_a_receive_that_finishes_within_the_bound();
        test_interrupt();
        test_a_replay_ends_by_itself();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL (%s): %s\n", check::current_step.load(), e.what());
        return 1;
    }
    std::error_code ec;
    fs::remove_all(g_tmp, ec);
    return check::report("qrss_listen_api");
}
