// Speculative latent optimization: run it while the operator is still
// composing, so the wait is mostly gone by the time they press Send.
//
// The measured gain curve is steep early and flat late (65% by 20 s,
// plateau ~90 s), and almost all of that can be spent in time the
// operator was using anyway. So: a short debounce after the picture
// stops changing, then optimize to plateau or a generous budget; if
// Send arrives first, a second and much shorter budget starts from the
// click. See docs/latent-optimization.md.
//
// **Qt-free and onnxruntime-free.** The gradient arrives through a
// factory, so this whole class -- the debounce, the generation
// counter, the deadline policy, the staleness rule -- is testable with
// a stub in a `--no-codec` build. That matters more here than
// elsewhere: the failure this code exists to prevent is transmitting
// the *previous* composition, which no amount of looking at a GUI
// reliably catches.

#ifndef SSTVAE_OPTIMIZE_SPECULATIVE_HPP
#define SSTVAE_OPTIMIZE_SPECULATIVE_HPP

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "config.hpp"
#include "images/types.hpp"
#include "optimize/optimize.hpp"

namespace sstvae::optimize {

// Given the picture to optimize toward, produce a gradient function
// bound to it. The real one makes a `codec::GradSession`; a test makes
// arithmetic. Called on the worker thread.
using GradFactory = std::function<GradFn(const images::ImageArray& target)>;

// The encoder's latents for the picture, produced on demand.
//
// A function rather than a vector so the encode happens on the worker
// *after* the debounce: it is ~30 ms of onnxruntime, which has no
// business on the GUI thread, and an edit that is superseded while the
// debounce is running never pays for it at all.
using LatentsFn = std::function<std::vector<double>()>;

struct SpeculativeConfig {
    // How long after the last edit to wait before starting. Long
    // enough that dragging an overlay item does not start a run per
    // mouse move; short enough to be underway before Send.
    double debounce_s = 1.0;

    // The budget when nobody is waiting. Generous on purpose -- this
    // is time the operator is spending anyway, and the plateau test
    // normally ends the run well before it.
    double idle_budget_s = 120.0;

    // The budget once Send has been pressed, measured from the click.
    // A floor of about one step (~1 s at four channel draws) applies
    // whatever this says, so values below that are not meaningful.
    double send_budget_s = 8.0;

    Options options{};
};

// What the GUI paints. A value copy, taken under the lock.
struct SpeculativeStatus {
    std::uint64_t generation = 0;
    bool idle = true;        // nothing scheduled or running
    bool waiting = false;    // in the debounce
    bool running = false;
    bool finished = false;   // a result for `generation` is ready
    bool send_pending = false;
    Progress progress{};     // last reported step of the current run
    StopReason stop = StopReason::MaxSteps;  // meaningful once finished
};

class Speculative {
public:
    // `on_change` is called (on the worker thread) whenever the status
    // moves, so a GUI can queue a repaint rather than poll. May be
    // null.
    Speculative(GradFactory factory, SpeculativeConfig config,
                std::function<void()> on_change = nullptr);
    ~Speculative();

    Speculative(const Speculative&) = delete;
    Speculative& operator=(const Speculative&) = delete;

    // The picture (or mode) changed. Abandons any run in flight, bumps
    // the generation, and schedules a fresh one after the debounce.
    //
    // `latents` produces the encoder's output for `target`, and is
    // called on the worker after the debounce -- this class does not
    // own a codec, it just needs a starting point.
    void picture_changed(images::ImageArray target, LatentsFn latents,
                         const config::ModeSpec& mode);

    // No picture to work on (the composition was cleared). Abandons
    // anything in flight.
    void clear();

    // Send was pressed: shorten the deadline. Non-blocking, and safe
    // whether the run is finished, running, still in the debounce, or
    // never started.
    void request_send();

    // The latents to transmit for the current generation, or empty if
    // none are ready yet. Never stale: a result from a superseded
    // generation is dropped rather than returned.
    //
    // Always at least as good as the encoder's own output -- `run`
    // returns its best iterate, and the first one it measures is the
    // input -- so there is no path where enabling this makes a picture
    // worse.
    std::vector<double> take_result() const;

    // True once `take_result` would return something for the current
    // generation, or once there is nothing to wait for. The GUI keys
    // "start transmitting" off this after `request_send`.
    bool ready() const;

    SpeculativeStatus status() const;

    // Stop the worker. Called by the destructor; safe to call twice.
    void stop();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace sstvae::optimize

#endif
