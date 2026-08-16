// The decoder's gradient graph, as an `optimize::GradFn`.
//
// This is the onnxruntime half of transmit-time latent optimization;
// the loop that uses it is in `optimize/optimize.hpp` and deliberately
// does not link ORT. The published artifact's forward pass *is* the
// backward pass -- the decoder's input-gradient written out as ordinary
// tensor ops and exported (see `sstvae/models/decoder_vjp.py`) -- so
// nothing here differentiates anything: it is one `Run()` per call.
//
// The MSE loss lives inside the graph, which is why a target picture
// has to be bound before the gradient can be asked for. Taking
// d(loss)/d(recon) as an input would keep the graph loss-agnostic, but
// no loss of the reconstruction can be evaluated before the
// reconstruction exists, so the caller would run the graph once for
// `recon` and again for the gradient -- doubling the cost of every step.
//
// **The target is set separately from construction, and that split is
// the point.** Building this loads a ~9 MB artifact from disk and
// stands up an `Ort::Env` and an `Ort::Session`; the target is one
// tensor. Binding the target in the constructor made the two
// inseparable, so `Speculative`'s factory built a whole new session for
// every picture -- and it is asked for a new one on every edit and, if
// the composition carries a "last received" inset, on every reception.
// One session per optimizer, re-targeted per run, is what that shape
// should have been.

#ifndef SSTVAE_CODEC_GRAD_SESSION_HPP
#define SSTVAE_CODEC_GRAD_SESSION_HPP

#include <memory>
#include <string>

#include "images/types.hpp"
#include "optimize/optimize.hpp"

namespace sstvae::codec {

class GradSession {
public:
    // `artifact` is a path to a `*-decoder-grad-fp32.onnx`; get one from
    // `checkpoint::resolve_onnx(checkpoint::GRAD_PART, ...)`, which pins
    // the precision because fp32 is the only one published.
    explicit GradSession(const std::string& artifact);
    ~GradSession();
    GradSession(GradSession&&) noexcept;
    GradSession& operator=(GradSession&&) noexcept;

    // The picture to optimize toward. Must be called before `fn()`, and
    // may be called again to reuse this session for another picture --
    // which is the whole reason it is not a constructor argument.
    void set_target(const images::ImageArray& target);

    // Suitable for `optimize::run`. The returned function borrows this
    // object, so it must not outlive it -- and it reads the target at
    // call time, so re-targeting invalidates any function still in
    // flight. Both are satisfied by the one caller, which runs the loop
    // to completion on a single worker before asking for the next.
    optimize::GradFn fn();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace sstvae::codec

#endif
