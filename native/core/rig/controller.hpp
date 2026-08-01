// Rig control that can never block the thread drawing the window.
//
// The reference was `sstvae/gui/rig_controller.py` (deleted 2026-08-01
// with the Python GUI), but this is a re-derivation rather than a port,
// and `docs/native-app.md` ("Bundling Hamlib") says why: Python talked
// to a `rigctld` child over a socket *because the SWIG Hamlib bindings
// live in the system site-packages and a virtualenv cannot see them*.
// C++ has no such problem, so the constraint that produced that
// architecture is gone. `libhamlib` is linked in-process and the socket
// client, the redial logic, the `rigctld` spawner and the `rigctld -l`
// column parser were never translated.
//
// **The property that survives unchanged is the one that matters**:
// nothing on the GUI thread ever blocks on the rig, and keying is never
// stuck behind a stale poll. Every rig call is blocking I/O that may
// take the rig's full timeout, so:
//
//   * **One backend, owned by one worker thread.** Every operation is a
//     job submitted to it. The handle is touched from nowhere else,
//     which also sidesteps a `RIG*` not being thread-safe.
//   * **PTT is a priority job, and submitting it discards pending
//     polls.** A queued poll is stale by definition -- its answer is a
//     frequency readout. Worst-case keying latency is therefore *one
//     in-flight operation*, never a queue drain. This replaces the
//     reference's separate-PTT-socket trick, which existed to dodge
//     contention that the socket layer itself introduced.
//   * **Polling suspends while transmitting.** The app is already half
//     duplex, so in the normal case PTT contends with nothing at all.
//   * **`stop()` detaches; it never joins.** A worker stuck in a
//     blocking serial read is abandoned, and its exit path owns closing
//     the handle. That is expressed here by the worker co-owning its
//     `Session` through a `shared_ptr`: `stop()` drops the controller's
//     reference, and whichever side is last -- almost always the
//     departing thread -- runs the destructor that closes the backend.
//     Joining would inherit the timeout being escaped.
//   * **A superseded worker cannot publish.** Callbacks hang off the
//     session, and a stopped session's are never invoked, so a reply
//     that arrives after reconfiguration cannot overwrite the new
//     worker's frequency or status. The reference achieves this by
//     passing locals into the thread; same idea, enforced by ownership.
//
// Qt-free, like the rx and tx engines: the GUI adapts the callbacks to
// signals. Blocking calls (`set_ptt`) are for worker threads only --
// `TxEngine` already runs on one.

#ifndef SSTVAE_RIG_CONTROLLER_HPP
#define SSTVAE_RIG_CONTROLLER_HPP

#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

#include "rig/backend.hpp"

namespace sstvae::rig {

// Everything the worker thread touches, co-owned by it and the
// controller. Defined in the .cpp; at namespace scope rather than
// nested so the worker function can name it without being a member.
struct RigSession;

// A rig that stops answering rarely starts again within one interval,
// and polling at full rate turns a misconfigured setup into a permanent
// stream of multi-second timeouts.
inline constexpr double MAX_BACKOFF_S = 60.0;

// How long to wait before the next poll: back to the configured rate on
// success, doubling up to the cap on failure.
double next_interval(double current, double base, bool ok);

struct RigConfig {
    double poll_interval_s = 5.0;
    // Ceiling on a single blocking operation, so `set_ptt` reports
    // rather than hanging the transmit sequence. The backend's own
    // timeout should be shorter; this is the backstop for when it is
    // not honoured.
    double operation_timeout_s = 10.0;
};

// Called from the worker thread, never from the caller's.
using OnFrequency = std::function<void(std::optional<double> hz)>;
using OnStatus = std::function<void(const std::string& text)>;

class RigController {
public:
    RigController(OnFrequency on_frequency = {}, OnStatus on_status = {});
    ~RigController();

    RigController(const RigController&) = delete;
    RigController& operator=(const RigController&) = delete;

    // Take ownership of a backend and start polling. Safe to call
    // repeatedly; supersedes any previous session.
    void start(std::unique_ptr<RigBackend> backend, const RigConfig& config = {});

    // Tear the session down. **Returns without waiting for the worker**,
    // so it is safe from the GUI thread even against a wedged rig.
    void stop();

    // Whether a session is configured -- **not** whether the rig is
    // answering. A worker whose open() failed has already exited while
    // this still reports true, exactly as the reference's thread does.
    // Rig health is reported through the status callback, which is the
    // only thing that can distinguish "connecting" from "no such
    // device" anyway.
    bool running() const;

    // Wait until an abandoned worker has finished shutting the rig down.
    // Returns true if it has; false if `seconds` elapsed first.
    //
    // Deliberately separate from `stop()`, which must never wait --
    // that is the guarantee this whole class exists for. This is for
    // the one caller that genuinely can afford to wait and must:
    // whatever is about to end the process.
    //
    // **Ending the process with a worker still inside libhamlib is a
    // real hazard on Windows**, where teardown takes the loader lock
    // and a thread cannot exit while it is held -- so the `pthread_join`
    // inside `rig_close` can block forever. Linux and macOS have no
    // equivalent, which is exactly the kind of asymmetry that shows up
    // as one platform's CI hanging.
    bool wait_for_shutdown(double seconds = 5.0);

    // The last frequency the worker published, or nothing. A cached
    // value, never a request -- reading it cannot block.
    std::optional<double> frequency_hz() const;

    // Key or unkey, blocking until the rig has actually been told.
    //
    // Blocking on purpose: `TxEngine` needs the rig keyed *before* it
    // waits out `ptt_lead_s`, so an asynchronous version would make that
    // delay measure the wrong interval. Call it from the transmit
    // worker, never from a UI thread. Throws RigError if there is no
    // session, if the rig refuses, or if the operation outlives
    // `operation_timeout_s`.
    void set_ptt(bool on);

    // A `tx::Ptt` for `TxEngine`. Keeps working across `start`/`stop`:
    // keying with no session throws RigError, which the engine reports
    // and treats as a failure to key -- the right answer for a
    // transmission that cannot key the radio.
    std::function<void(bool)> ptt_function();

    // Transmit interlock. Polling stops while keyed: the answer is not
    // interesting mid-over, and some rigs dislike CAT traffic while
    // transmitting.
    void pause_polling();
    void resume_polling();

private:
    std::shared_ptr<RigSession> session_;
    // Kept after stop() purely to observe the worker's teardown; a weak
    // reference, so holding it cannot keep the session alive.
    std::weak_ptr<RigSession> departing_;
    OnFrequency on_frequency_;
    OnStatus on_status_;
    mutable std::mutex mutex_;
};

}  // namespace sstvae::rig

#endif
