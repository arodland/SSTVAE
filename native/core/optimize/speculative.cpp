#include "optimize/speculative.hpp"

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>

namespace sstvae::optimize {

namespace {
using Clock = std::chrono::steady_clock;

Clock::duration secs(double s) {
    return std::chrono::duration_cast<Clock::duration>(
        std::chrono::duration<double>(s));
}
}  // namespace

struct Speculative::Impl {
    GradFactory factory;
    SpeculativeConfig cfg;
    std::function<void()> on_change;

    mutable std::mutex m;
    std::condition_variable cv;
    bool quit = false;

    // Bumped by every edit. A run carries the generation it started
    // for and abandons itself the moment that stops matching, which is
    // the whole point of this class: a result belongs to one picture,
    // and shipping a superseded one would transmit the composition the
    // operator just changed.
    std::uint64_t generation = 0;

    // The job waiting for its debounce to expire, if any.
    struct Job {
        std::uint64_t generation = 0;
        images::ImageArray target;
        LatentsFn latents;
        const config::ModeSpec* mode = nullptr;
        Clock::time_point start_after{};
    };
    std::optional<Job> pending;

    // Set by request_send, cleared by a new generation. Absolute, so
    // the deadline is measured from the *click* rather than from
    // wherever the run happens to be.
    std::optional<Clock::time_point> send_at;

    SpeculativeStatus status;
    std::uint64_t result_generation = 0;
    std::vector<double> result;

    std::thread worker;

    // The callback runs *without* the lock, so a GUI is free to take
    // its own -- and it takes the lock by reference rather than
    // unlocking the mutex directly, which would leave the caller's
    // unique_lock believing something false about its own state.
    void notify(std::unique_lock<std::mutex>& lock) {
        if (!on_change) return;
        auto fn = on_change;
        lock.unlock();
        fn();
        lock.lock();
    }

    void run();
};

void Speculative::Impl::run() {
    std::unique_lock<std::mutex> lock(m);
    while (!quit) {
        if (!pending) {
            cv.wait(lock);
            continue;
        }
        // Wait out the debounce, but wake for a new edit, a Send, or
        // shutdown. Send during the debounce starts the run
        // immediately -- there is no reason to keep waiting for an
        // editor that is no longer being used.
        const Clock::time_point until = pending->start_after;
        if (Clock::now() < until && !send_at && !quit) {
            cv.wait_until(lock, until);
            continue;
        }

        Job job = *pending;
        pending.reset();
        status.waiting = false;
        status.running = true;
        status.idle = false;
        status.finished = false;
        status.progress = Progress{};
        notify(lock);

        const std::uint64_t mine = job.generation;
        GradFactory factory_copy = factory;
        SpeculativeConfig cfg_copy = cfg;

        lock.unlock();

        // Everything below runs unlocked: building the session loads a
        // model, and `run` is seconds of arithmetic.
        Result result;
        std::vector<double> start;
        try {
            start = job.latents();
            GradFn grad = factory_copy(job.target);
            Options opts = cfg_copy.options;
            opts.time_budget_s = cfg_copy.idle_budget_s;
            const Clock::time_point run_start = Clock::now();

            result = optimize::run(
                grad, start, *job.mode, opts,
                [&](const Progress& p) {
                    std::function<void()> notify_fn;
                    bool keep_going = true;
                    {
                        std::lock_guard<std::mutex> g(m);
                        // Stale or shutting down: abandon without
                        // publishing anything. This is the check the
                        // whole class exists for.
                        if (quit || generation != mine) return false;
                        status.progress = p;
                        notify_fn = on_change;

                        // The deadline that binds right now. Before
                        // Send that is the idle budget; after it,
                        // whichever of the two expires first -- and it
                        // is measured from the *click*, not from here,
                        // so a run that was nearly done still stops
                        // promptly.
                        const Clock::time_point now = Clock::now();
                        if (send_at &&
                            now >= *send_at + secs(cfg_copy.send_budget_s)) {
                            keep_going = false;
                        }
                        if (now - run_start >= secs(cfg_copy.idle_budget_s)) {
                            keep_going = false;
                        }
                    }
                    // One repaint per ~1 s of arithmetic, outside the
                    // lock.
                    if (notify_fn) notify_fn();
                    return keep_going;
                });
        } catch (...) {
            // A missing or broken artifact must not take the app down,
            // and must not leave the operator unable to transmit: the
            // encoder's latents are still perfectly good.
            lock.lock();
            status.running = false;
            if (generation == mine) {
                // Fall back to the encoder's own latents rather than
                // leaving the operator unable to transmit. A missing or
                // broken artifact costs the improvement, not the
                // picture.
                status.finished = true;
                status.idle = true;
                result_generation = mine;
                // `start` is empty if the *encode* was what failed, in
                // which case there is nothing to fall back to and the
                // caller has to encode for itself as it always did.
                this->result = start;
                status.stop = StopReason::Cancelled;
                notify(lock);
            }
            continue;
        }

        lock.lock();
        status.running = false;
        if (generation == mine) {
            result_generation = mine;
            this->result = std::move(result.latents);
            status.finished = true;
            status.idle = true;
            status.stop = result.stop;
            notify(lock);
        }
        // If the generation moved on, `pending` already holds the
        // replacement and the loop picks it up without publishing
        // anything for the abandoned one.
    }
}

Speculative::Speculative(GradFactory factory, SpeculativeConfig config,
                         std::function<void()> on_change)
    : impl_(std::make_unique<Impl>()) {
    impl_->factory = std::move(factory);
    impl_->cfg = config;
    impl_->on_change = std::move(on_change);
    impl_->worker = std::thread([this] { impl_->run(); });
}

Speculative::~Speculative() { stop(); }

void Speculative::stop() {
    if (!impl_) return;
    {
        std::lock_guard<std::mutex> g(impl_->m);
        if (impl_->quit) return;
        impl_->quit = true;
    }
    impl_->cv.notify_all();
    if (impl_->worker.joinable()) impl_->worker.join();
}

void Speculative::picture_changed(images::ImageArray target, LatentsFn latents,
                                  const config::ModeSpec& mode) {
    {
        std::lock_guard<std::mutex> g(impl_->m);
        ++impl_->generation;
        impl_->send_at.reset();
        // The previous result is deliberately *not* cleared here. The
        // generation counter is the single mechanism that decides
        // whether a result may be handed out, and having two -- a
        // clear here as well as the check in `take_result` -- would
        // make the check unreachable, which is to say untested. Anyone
        // removing it should see a failure.

        Impl::Job job;
        job.generation = impl_->generation;
        job.target = std::move(target);
        job.latents = std::move(latents);
        job.mode = &mode;
        job.start_after = Clock::now() + secs(impl_->cfg.debounce_s);
        impl_->pending = std::move(job);

        impl_->status.generation = impl_->generation;
        impl_->status.idle = false;
        impl_->status.waiting = true;
        impl_->status.running = false;
        impl_->status.finished = false;
        impl_->status.send_pending = false;
        impl_->status.progress = Progress{};
    }
    impl_->cv.notify_all();
}

void Speculative::clear() {
    {
        std::lock_guard<std::mutex> g(impl_->m);
        ++impl_->generation;
        impl_->pending.reset();
        impl_->send_at.reset();
        impl_->result.clear();
        impl_->result_generation = 0;
        impl_->status = SpeculativeStatus{};
        impl_->status.generation = impl_->generation;
    }
    impl_->cv.notify_all();
}

void Speculative::request_send() {
    {
        std::lock_guard<std::mutex> g(impl_->m);
        if (!impl_->send_at) impl_->send_at = Clock::now();
        impl_->status.send_pending = true;
    }
    impl_->cv.notify_all();
}

std::vector<double> Speculative::take_result() const {
    std::lock_guard<std::mutex> g(impl_->m);
    if (impl_->result_generation != impl_->generation) return {};
    return impl_->result;
}

bool Speculative::ready() const {
    std::lock_guard<std::mutex> g(impl_->m);
    // "Nothing to wait for" counts as ready: with no picture scheduled
    // the caller has nothing to collect and should not be blocked.
    if (!impl_->pending && !impl_->status.running && !impl_->status.finished) {
        return true;
    }
    return impl_->result_generation == impl_->generation;
}

SpeculativeStatus Speculative::status() const {
    std::lock_guard<std::mutex> g(impl_->m);
    return impl_->status;
}

}  // namespace sstvae::optimize
