// The live QRSSTVAE listener as a library (listen_api.hpp says what and why).

#include "qrss/listen_api.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <mutex>
#include <string_view>
#include <thread>
#include <utility>

#include "qrss/config.hpp"
#include "qrss/dirlock.hpp"
#include "qrss/frame.hpp"
#include "qrss/passband.hpp"
#include "qrss/pypath.hpp"
#include "qrss/pytext.hpp"
#include "qrss/receiver_slot.hpp"
#include "qrss/rules.hpp"
#include "qrss/rx_api.hpp"
#include "qrss/store.hpp"

namespace sstvae::qrss::listen {

namespace {

using steady = std::chrono::steady_clock;
namespace fr = sstvae::qrss::frame;

constexpr int REQ_NONE = 0;
constexpr int REQ_END = 1;
constexpr int REQ_INTERRUPT = 2;

// What the session's thread throws out of live::run to leave it without
// writing anything (Abandoned) or for the script's KeyboardInterrupt
// (Interrupted). Not std::exceptions: nothing on the way may take them for
// an error.
struct Interrupted {};
struct Abandoned {};

steady::duration seconds(double s) {
    return std::chrono::duration_cast<steady::duration>(std::chrono::duration<double>(s));
}

// The name a stage of an abandoned receive fails with.
live::SeamError stopped() {
    return live::SeamError("ListenerStopped", "the listener stopped while this was being received");
}

// The parts of a session its wrapped seams share with it (and only these: a
// seam that held the session would keep it alive through its listener).
struct Guard {
    std::atomic<bool> abandoned{false};

    // the sink: one line at a time, none once abandoned
    std::mutex sink_mu;
    Sink sink;
    void emit(LineKind kind, const std::string& text) {
        std::lock_guard lk(sink_mu);
        if (abandoned.load()) return;
        try {
            if (sink) {
                sink(kind, text);
            } else {
                std::fputs((text + "\n").c_str(), stderr);
                std::fflush(stderr);
            }
        } catch (...) {
            // a sink that throws must not take the listener's thread down
        }
    }

    // associations under way, and the store's lock
    std::mutex mu;
    std::condition_variable cv;
    int associating = 0;
    std::optional<dirlock::DirLock> store_lock;
    bool release_store = false;   // the session has let go: the last association releases it

    void store_released_if_due() {   // (mu held)
        if (release_store && associating == 0 && store_lock) store_lock.reset();
    }
};

// The listener's pictures, through the guard: none drawn once abandoned (the
// listener writes none then either: LiveListener::abandon).
class GuardedRenderer final : public live::Renderer {
public:
    GuardedRenderer(std::shared_ptr<live::Renderer> inner, std::shared_ptr<Guard> g)
        : inner_(std::move(inner)), g_(std::move(g)) {}

    std::optional<std::string> load(std::optional<std::int64_t> codec_id, const Warn& warn) override {
        if (g_->abandoned.load()) return std::nullopt;   // render() refuses below
        return inner_->load(codec_id, warn);
    }
    std::optional<std::string> check(std::optional<std::int64_t> codec_id) override {
        if (g_->abandoned.load()) return std::nullopt;
        return inner_->check(codec_id);
    }
    images::Picture render(std::span<const double> S, std::span<const double> W, std::int64_t mode,
                           std::optional<std::int64_t> codec_id) override {
        if (g_->abandoned.load()) throw stopped();
        images::Picture pic = inner_->render(S, W, mode, codec_id);
        if (g_->abandoned.load()) throw stopped();
        return pic;
    }

private:
    std::shared_ptr<live::Renderer> inner_;
    std::shared_ptr<Guard> g_;
};

// The listener's seams with every receive stage and the store's association
// refused once the session has abandoned them. An empty seam becomes the
// listener's own default (live::default_seams, the one copy) so that it can
// be wrapped.
live::Seams guarded_seams(live::Seams s, const live::LiveConfig& cfg, const std::shared_ptr<Guard>& g) {
    s = live::default_seams(cfg, std::move(s));

    // each stage: refused before it starts, and its result dropped where it
    // returns, once abandoned
    auto check = [g] {
        if (g->abandoned.load()) throw stopped();
    };
    s.candidates = [f = std::move(s.candidates), check](const receiver::Prepared& prep, live::Selection sel,
                                                        std::span<const acquire::Hint> hints) {
        check();
        auto r = f(prep, sel, hints);
        check();
        return r;
    };
    s.detect = [f = std::move(s.detect), check](const receiver::Prepared& prep, live::Selection sel,
                                                const receiver::DetectOptions& opt) {
        check();
        auto r = f(prep, sel, opt);
        check();
        return r;
    };
    s.receive_slot = [f = std::move(s.receive_slot), check](const receiver::Prepared& prep, live::Selection sel,
                                                            const receiver::SlotOptions& opt) {
        check();
        auto r = f(prep, sel, opt);
        check();
        return r;
    };
    s.finish = [f = std::move(s.finish), check](std::span<const live::cf> x, std::span<const std::uint8_t> ok,
                                                std::int64_t origin, std::int64_t q, const api::ReceiveConfig& config,
                                                std::span<const types::Detection> dets,
                                                std::span<const acquire::Hint> hints) {
        check();
        auto r = f(x, ok, origin, q, config, dets, hints);
        check();
        return r;
    };
    // the association writes the store: counted, so the session can wait for
    // the one under way before it lets the store's lock go
    s.associate = [f = std::move(s.associate), g](const types::PassResult& p, store::Store& st) {
        {
            std::lock_guard lk(g->mu);
            if (g->abandoned.load()) throw stopped();
            ++g->associating;
        }
        struct Done {
            Guard& g;
            ~Done() {
                {
                    std::lock_guard lk(g.mu);
                    --g.associating;
                    g.store_released_if_due();
                }
                g.cv.notify_all();
            }
        } done{*g};
        return f(p, st);
    };
    return s;
}

// argparse's choices refusal, for an Options holding what the parser would not have taken.
std::string invalid_choice(const std::string& flag, const std::string& value,
                           std::initializer_list<const char*> choices) {
    std::string out = "argument " + flag + ": invalid choice: " + pytext::repr(std::string_view(value)) +
                      " (choose from ";
    bool first = true;
    for (const char* c : choices) {
        out += (first ? "" : ", ") + pytext::repr(std::string_view(c));
        first = false;
    }
    return out + ")";
}

// ', '.join of a preset table's names, sorted or in its order.
std::string names(config::PresetGroup group, bool sorted) {
    std::vector<std::string> v;
    for (const fr::FrameSpec& s : fr::presets(group)) v.push_back(s.name());
    if (sorted) std::sort(v.begin(), v.end());
    std::string out;
    for (std::size_t k = 0; k < v.size(); ++k) out += (k ? ", " : "") + v[k];
    return out;
}

}  // namespace

// --- configuration ------------------------------------------------------------------------------

live::LiveConfig build_config(const Options& opt) {
    if (opt.just_ce && opt.just_l) throw Refusal("argument --just-l: not allowed with argument --just-ce");
    if (opt.precision != "fp32" && opt.precision != "fp16" && opt.precision != "int8")
        throw Refusal(invalid_choice("--precision", opt.precision, {"fp32", "fp16", "int8"}));
    // the frame as sys.argv holds it: a byte that is not UTF-8 is an escape there
    const std::string shown = pytext::repr(pypath::fs_decode(opt.frame));
    std::optional<fr::FrameSpec> spec;
    try {   // (an L name, as qrss_receive.py takes, names its CE twin)
        spec = fr::twin(fr::get_any(pypath::fs_str(opt.frame)), config::WAVEFORM_CE);
    } catch (const std::invalid_argument&) {
        throw Refusal("--frame " + shown + ": one of " + names(config::PresetGroup::Presets, true) + ", " +
                      names(config::PresetGroup::LPresets, true));
    } catch (const std::out_of_range&) {
        throw Refusal("--frame " + shown + ": one of " + names(config::PresetGroup::Presets, true) + ", " +
                      names(config::PresetGroup::LPresets, true));
    }
    std::vector<std::int64_t> waveforms;
    if (opt.just_ce) waveforms = {config::WAVEFORM_CE};
    else if (opt.just_l) waveforms = {config::WAVEFORM_L};
    else waveforms = {config::WAVEFORM_CE, config::WAVEFORM_L};
    if (std::find(waveforms.begin(), waveforms.end(), std::int64_t{config::WAVEFORM_L}) != waveforms.end()) {
        try {
            (void)fr::twin(*spec, config::WAVEFORM_L);
        } catch (const std::invalid_argument&) {
            throw Refusal("--frame " + shown +
                          " is CE only: waveform L has no frame of that shape (L frames: " +
                          names(config::PresetGroup::LPresets, false) + "); add --just-ce to receive it");
        }
    }
    if (opt.refresh_s <= 0) throw Refusal("--refresh must be positive");
    live::LiveConfig cfg;
    cfg.spec = *spec;
    cfg.waveforms = waveforms;
    cfg.refresh_s = opt.refresh_s;
    cfg.precision = opt.precision;
    cfg.model = opt.model;
    cfg.render = opt.render;
    cfg.background = true;
    cfg.finish_process = true;
    return cfg;
}

std::string store_root(const Options& opt) {
    // Path(args.store) if args.store else default_root(): "" is the default too
    if (opt.store_dir && !opt.store_dir->empty()) return pypath::pathlib_str(*opt.store_dir);
    return store::default_root();
}

std::string state_dir(const Options& opt) {
    // args.state if args.state is not None else root / "live"
    if (opt.state_dir) return pypath::pathlib_str(*opt.state_dir);
    return pypath::child(store_root(opt), "live");
}

// --- PushSource ----------------------------------------------------------------------------------

PushSource::PushSource(std::string name, live::Clock clock, double hold_s)
    : name_(std::move(name)), clock_(std::move(clock)), stamper_(hold_s) {}

bool PushSource::push(std::span<const float> x) {
    std::lock_guard lk(mu_);   // the clock and the stamper in arrival order
    if (ended_) return false;
    if (x.empty()) return true;   // a read of nothing is no read
    std::vector<double> v(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) v[i] = static_cast<double>(x[i]) * 1.0;   // StreamSource's f32
    try {
        const double t = clock_ ? clock_() : live::system_time();
        for (live::Chunk& c : stamper_.push(std::move(v), t)) q_.push_back(std::move(c));
    } catch (...) {
        // as StreamSource: what the clock or the Stamper throws ends the stream
        error_ = std::current_exception();
        try {
            for (live::Chunk& c : stamper_.flush()) q_.push_back(std::move(c));
        } catch (...) {
        }
        ended_ = true;
    }
    cv_.notify_all();
    return true;
}

void PushSource::end() {
    std::lock_guard lk(mu_);
    if (ended_) return;
    try {
        for (live::Chunk& c : stamper_.flush()) q_.push_back(std::move(c));
    } catch (...) {
        if (!error_) error_ = std::current_exception();
    }
    ended_ = true;
    cv_.notify_all();
}

std::optional<live::Chunk> PushSource::next(double timeout_s) {
    if (!(timeout_s >= 0)) throw std::invalid_argument("'timeout' must be a non-negative number");
    std::unique_lock lk(mu_);
    auto ready = [&] { return !q_.empty() || ended_; };
    if (timeout_s >= 1e9) cv_.wait(lk, ready);
    else cv_.wait_for(lk, std::chrono::duration<double>(timeout_s), ready);
    if (q_.empty()) return std::nullopt;
    live::Chunk c = std::move(q_.front());
    q_.pop_front();
    return c;
}

bool PushSource::eof() const {
    std::lock_guard lk(mu_);
    return ended_ && q_.empty();
}

std::exception_ptr PushSource::error() const {
    std::lock_guard lk(mu_);
    return error_;
}

std::size_t PushSource::backlog() const {
    std::lock_guard lk(mu_);
    std::size_t n = 0;
    for (const live::Chunk& c : q_) n += c.x.size();
    return n;
}

// --- the session ----------------------------------------------------------------------------------

struct Session::Core : std::enable_shared_from_this<Session::Core> {
    class Gate;

    Options opt;
    live::LiveConfig cfg;
    std::string state;
    std::optional<std::string> root;      // held
    std::optional<std::string> pb_dir;    // held
    double bound_s = STOP_BOUND_S;
    live::Clock clock;
    std::shared_ptr<Guard> guard = std::make_shared<Guard>();

    std::optional<dirlock::DirLock> state_lock;
    std::optional<dirlock::DirLock> pb_lock;   // (the store's is the guard's)
    std::shared_ptr<store::Store> store;
    std::shared_ptr<passband::PassbandStore> passband;
    std::shared_ptr<live::LiveListener> listener;
    std::unique_ptr<live::Source> inner;
    PushSource* push = nullptr;           // inner, when it is the session's own
    bool realtime = true;
    std::unique_ptr<Gate> gate;

    // The session's thread holds run_mu while it works and lets it go while
    // it waits for audio, so a stop that takes it knows that thread is
    // writing nothing; `finishing` says it is on its last writes.
    std::timed_mutex run_mu;
    std::atomic<int> req{REQ_NONE};
    std::atomic<bool> finishing{false};

    mutable std::mutex mu;
    mutable std::condition_variable cv;
    bool run_done = false;
    bool closer_started = false;
    bool closer_done = false;
    bool shut_started = false;
    bool shut_done = false;
    bool ended_done = false;   // shut_done, and on_ended returned
    bool was_interrupted = false;
    std::exception_ptr error;
    steady::time_point deadline{};
    std::thread run_thread;
    std::thread closer;

    ~Core() { let_go(); }

    double now() const { return clock(); }

    // A sleep the stop cuts short (a replay's pacing).
    void sleep(double s) {
        std::unique_lock lk(mu);
        if (!(s > 0)) return;
        cv.wait_for(lk, std::chrono::duration<double>(std::min(s, 1e6)), [&] { return req.load() != REQ_NONE; });
    }

    void request(int kind) {
        int cur = req.load();
        while (cur < kind && !req.compare_exchange_weak(cur, kind)) {
        }
        if (push) push->end();   // nothing more is taken
        {
            std::lock_guard lk(mu);
        }
        cv.notify_all();
    }

    void run_body();
    void start_closer();
    void abandon(bool on_run_thread);
    void release_locks();
    void shutdown(int kind, bool on_run_thread);
    void let_go();
};

// The session's source as live::run sees it: the inner one, polled in short
// slices so a stop is seen within one, with the session's thread's run_mu
// let go while it waits. After a stop's request: the end (stop) or
// Interrupted (interrupt) once the PushSource's queue, if it is one, has
// been read out; Abandoned once abandoned.
class Session::Core::Gate final : public live::Source {
public:
    explicit Gate(Core* c) : c_(c) {}
    void bind(std::unique_lock<std::timed_mutex>* held) { held_ = held; }

    std::optional<live::Chunk> next(double timeout_s) override {
        Core& c = *c_;
        std::optional<live::Chunk> got;
        bool forced = false;
        {
            held_->unlock();
            struct Relock {
                std::unique_lock<std::timed_mutex>* l;
                ~Relock() { l->lock(); }
            } relock{held_};
            const steady::time_point end =
                steady::now() + seconds(std::isfinite(timeout_s) ? std::clamp(timeout_s, 0.0, 1e6) : 1e6);
            for (;;) {
                const int r = c.req.load();
                if (r == REQ_INTERRUPT || c.guard->abandoned.load()) break;
                if (r == REQ_END && !c.push) {
                    forced = true;
                    break;
                }
                const double left = std::chrono::duration<double>(end - steady::now()).count();
                got = c.inner->next(std::clamp(left, 0.0, 0.05));
                if (got || c.inner->eof() || left <= 0) break;
            }
        }
        if (c.guard->abandoned.load()) throw Abandoned{};
        if (c.req.load() == REQ_INTERRUPT) {
            c.finishing = true;
            throw Interrupted{};
        }
        if (forced) {
            forced_eof_ = true;
            c.finishing = true;
            return std::nullopt;
        }
        if (!got && c.inner->eof()) c.finishing = true;
        return got;
    }
    bool eof() const override { return forced_eof_ || c_->inner->eof(); }
    std::string name() const override { return c_->inner->name(); }

private:
    Core* c_;
    std::unique_lock<std::timed_mutex>* held_ = nullptr;
    bool forced_eof_ = false;   // (the session's thread alone)
};

void Session::Core::run_body() {
    bool interrupted = false;
    std::exception_ptr failure;
    {
        std::unique_lock<std::timed_mutex> held(run_mu);
        gate->bind(&held);
        // (a stop that came first skips the read-back and the recovery: what
        // they would start, a stop this close behind would only abandon)
        auto stopping = [&] { return guard->abandoned.load() || req.load() != REQ_NONE; };
        try {
            if (realtime && !stopping()) listener->backfill(now());
            if (realtime && !stopping()) listener->recover(now(), opt.recover_hours);
            if (guard->abandoned.load()) throw Abandoned{};
            if (req.load() == REQ_INTERRUPT) {
                finishing = true;
                throw Interrupted{};
            }
            guard->emit(LineKind::notice, "tiles in " + state);
            live::run(*listener, *gate, realtime);
        } catch (const Interrupted&) {
            interrupted = true;
            try {
                if (!guard->abandoned.load()) {
                    listener->log("stopped");
                    listener->write_state(false);
                }
            } catch (...) {
                failure = std::current_exception();
            }
        } catch (const Abandoned&) {
        } catch (...) {
            failure = std::current_exception();
        }
    }
    {
        std::lock_guard lk(mu);
        run_done = true;
        was_interrupted = interrupted;
        error = failure;
    }
    cv.notify_all();
    shutdown(REQ_END, true);
}

void Session::Core::start_closer() {
    {
        std::lock_guard lk(mu);
        if (closer_started) return;
        closer_started = true;
    }
    // listener.close(): the workers' receives in flight (minutes, possibly),
    // then the packer. Its own thread, so a stop can give up on it.
    // (joined or detached by let_go; detached, it keeps the Core it signals)
    closer = std::thread([self = shared_from_this()] {
        try {
            self->listener->close();
        } catch (...) {
        }
        {
            std::lock_guard lk(self->mu);
            self->closer_done = true;
        }
        self->cv.notify_all();
    });
}

void Session::Core::abandon(bool on_run_thread) {
    {
        // The session's thread at a safe point (waiting for audio, or done),
        // else on its last writes, which are waited for (the time to write
        // state); anywhere else (reading the passband store back, or a step
        // that outlasts the bound) it is left where it is: the listener,
        // abandoned below, writes nothing more for it, and it stops at its
        // next read.
        std::unique_lock<std::timed_mutex> rl(run_mu, std::defer_lock);
        bool done;
        {
            std::lock_guard lk(mu);
            done = run_done;
        }
        if (!on_run_thread && !done) {
            if (!rl.try_lock_until(deadline) && finishing.load()) rl.lock();
        }
        // The listener first, while the session still holds every lock:
        // nothing more to listener.log, state.json, tiles/ or the passband
        // store (a write under way when this is called lands before it
        // returns), no job of it applies anything, and its packer starts no
        // further hour. It waits for no receive.
        listener->abandon();
        std::lock_guard g(guard->mu);
        guard->abandoned = true;
    }
    {
        // nothing more reaches the sink
        std::lock_guard s(guard->sink_mu);
    }
    guard->cv.notify_all();
    // an association under way: waited for to the deadline, else it lets the
    // store's lock go itself (release_locks)
    std::unique_lock g(guard->mu);
    guard->cv.wait_until(g, deadline, [&] { return guard->associating == 0; });
}

void Session::Core::release_locks() {
    state_lock.reset();
    pb_lock.reset();
    std::lock_guard g(guard->mu);
    guard->release_store = true;
    guard->store_released_if_due();
}

void Session::Core::shutdown(int kind, bool on_run_thread) {
    std::unique_lock lk(mu);
    if (shut_started) {
        if (!on_run_thread) cv.wait(lk, [&] { return shut_done; });
        return;
    }
    shut_started = true;
    deadline = steady::now() + seconds(bound_s);
    lk.unlock();
    request(kind);   // (after an error too: a PushSource takes nothing more)
    start_closer();
    lk.lock();
    const steady::time_point waits = deadline - seconds(std::min(SETTLE_S, bound_s / 4));
    if (!on_run_thread) cv.wait_until(lk, waits, [&] { return run_done; });
    cv.wait_until(lk, waits, [&] { return closer_done; });
    const bool clean = run_done && closer_done;
    const bool failed = error != nullptr;
    lk.unlock();
    if (!clean) {
        abandon(on_run_thread);
    } else if (!failed) {
        // What the close let finish (a receive in flight at the end of the
        // audio) landed after the session's thread wrote its last state:
        // written once more, so state.json shows the tiles listener.log and
        // the store already have (else the next start would neither list
        // them nor receive the slot again, its passes being stored).
        try {
            listener->write_state(false);
        } catch (...) {
        }
    }
    release_locks();
    lk.lock();
    shut_done = true;
    lk.unlock();
    cv.notify_all();
    if (opt.on_ended) {
        try {
            opt.on_ended();
        } catch (...) {
        }
    }
    lk.lock();
    ended_done = true;
    lk.unlock();
    cv.notify_all();
}

void Session::Core::let_go() {
    auto release = [&](std::thread& t, bool done) {
        if (!t.joinable()) return;
        if (t.get_id() == std::this_thread::get_id() || !done) t.detach();
        else t.join();
    };
    bool rd, cd;
    {
        std::lock_guard lk(mu);
        rd = run_done && shut_done;
        cd = closer_done;
    }
    release(run_thread, rd);
    release(closer, cd);
}

std::unique_ptr<Session> Session::start(Options opt, Sink sink) {
    auto c = std::make_shared<Core>();
    c->guard->sink = std::move(sink);
    c->cfg = build_config(opt);
    c->bound_s = std::isfinite(opt.stop_bound_s) && opt.stop_bound_s >= 0 ? opt.stop_bound_s : STOP_BOUND_S;
    c->clock = opt.clock ? opt.clock : live::Clock(live::system_time);

    // One listener per directory. A second one on the same tiles would
    // overwrite the first's state file, so it stops; on the same store it
    // runs without one (no combining with earlier passes, no passband).
    const std::string root = listen::store_root(opt);
    c->state = listen::state_dir(opt);
    c->state_lock = dirlock::DirLock::try_acquire(c->state);
    if (!c->state_lock)
        throw Refusal("another listener is writing tiles to " + c->state +
                      "; stop it, or give this one its own --state DIR");
    if (!opt.no_store) {
        std::optional<dirlock::DirLock> l = dirlock::DirLock::try_acquire(root);
        if (!l) {
            c->guard->emit(LineKind::notice, "another listener is using the store in " + root +
                                                 "; this one runs without a store (pass --store DIR for one "
                                                 "of its own)");
        } else {
            c->guard->store_lock = std::move(l);
            c->store = std::make_shared<store::Store>(root);
            c->root = c->store->root();
        }
    }
    if (c->store && !opt.no_passband) {
        const std::string pb = opt.passband_dir && !opt.passband_dir->empty()
                                   ? pypath::pathlib_str(*opt.passband_dir)
                                   : pypath::child(c->store->root(), "passband");
        std::optional<dirlock::DirLock> l = dirlock::DirLock::try_acquire(pb);
        if (!l) {
            c->guard->emit(LineKind::notice,
                           "another listener is writing the passband store in " + pb + "; this one keeps none");
        } else {
            c->pb_lock = std::move(l);
            passband::StoreOptions po;
            po.pack = true;
            c->passband = std::make_shared<passband::PassbandStore>(pb, po);
            c->pb_dir = pb;
        }
    }

    // rules.apply(args), and the environment's switches as rules.old reads them
    live::Seams s = opt.seams;
    s.clock = c->clock;
    s.old_edge_noise = opt.old_edge_noise || rules::old(rules::EDGE_NOISE);
    s.old_gating = opt.old_gating || rules::old(rules::GATING);
    std::shared_ptr<Guard> g = c->guard;
    s.log = [g](const std::string& line) { g->emit(LineKind::log, line); };
    bool no_renderer = false;
    s.renderer.reset();
    if (c->cfg.render) {
        std::shared_ptr<live::Renderer> r = opt.renderer ? opt.renderer(c->cfg) : nullptr;
        if (r) {
            s.renderer = std::make_shared<GuardedRenderer>(std::move(r), g);
        } else {
            no_renderer = true;
            c->cfg.render = false;   // (config() says what the session does)
        }
    }
    s = guarded_seams(std::move(s), c->cfg, g);
    c->listener = std::make_shared<live::LiveListener>(c->state, c->cfg, c->store, c->passband, std::move(s));

    Core* self = c.get();
    if (opt.source) {
        SourceContext ctx;
        ctx.clock = c->clock;
        ctx.sleep = [self](double sec) { self->sleep(sec); };
        c->inner = opt.source(ctx);
        if (!c->inner) throw std::invalid_argument("listen: the source factory made no source");
        c->realtime = opt.realtime;
    } else {
        auto p = std::make_unique<PushSource>(opt.push_name, c->clock, opt.push_hold_s);
        c->push = p.get();
        c->inner = std::move(p);
        c->realtime = true;
    }
    if (no_renderer) c->guard->emit(LineKind::notice, NO_PICTURES_NOTICE);
    c->opt = std::move(opt);
    c->gate = std::make_unique<Core::Gate>(self);
    std::shared_ptr<Core> keep = c;
    c->run_thread = std::thread([keep] { keep->run_body(); });
    return std::unique_ptr<Session>(new Session(std::move(c)));
}

Session::~Session() {
    try {
        c_->shutdown(REQ_END, false);
    } catch (...) {
    }
    c_->let_go();
}

bool Session::push(std::span<const float> x) { return c_->push ? c_->push->push(x) : false; }

std::size_t Session::backlog() const { return c_->push ? c_->push->backlog() : 0; }

void Session::stop() { c_->shutdown(REQ_END, false); }

void Session::interrupt() { c_->shutdown(REQ_INTERRUPT, false); }

bool Session::running() const { return !ended(); }

bool Session::ended() const {
    std::lock_guard lk(c_->mu);
    return c_->ended_done;
}

bool Session::wait_ended(std::optional<double> timeout_s) const {
    std::unique_lock lk(c_->mu);
    auto done = [&] { return c_->ended_done; };
    if (!timeout_s || !std::isfinite(*timeout_s)) {
        c_->cv.wait(lk, done);
        return true;
    }
    return c_->cv.wait_for(lk, std::chrono::duration<double>(std::max(0.0, *timeout_s)), done);
}

bool Session::interrupted() const {
    std::lock_guard lk(c_->mu);
    return c_->was_interrupted;
}

bool Session::abandoned() const { return c_->guard->abandoned.load(); }

std::exception_ptr Session::error() const {
    std::lock_guard lk(c_->mu);
    return c_->error;
}

const std::string& Session::state_dir() const { return c_->state; }
std::optional<std::string> Session::store_root() const { return c_->root; }
std::optional<std::string> Session::passband_dir() const { return c_->pb_dir; }
std::shared_ptr<store::Store> Session::store() const { return c_->store; }
std::shared_ptr<passband::PassbandStore> Session::passband() const { return c_->passband; }
const live::LiveConfig& Session::config() const { return c_->cfg; }
std::vector<live::Tile> Session::tiles() const { return c_->listener->tiles(); }
std::vector<std::string> Session::lines() const { return c_->listener->lines(); }

}  // namespace sstvae::qrss::listen
