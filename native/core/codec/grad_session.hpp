// The decoder's gradient graph, as an `optimize::GradFn`.
//
// This is the onnxruntime half of transmit-time latent optimization;
// the loop that uses it is in `optimize/optimize.hpp` and deliberately
// does not link ORT. The published artifact's forward pass *is* the
// backward pass -- the decoder's input-gradient written out as ordinary
// tensor ops and exported (see `sstvae/models/decoder_vjp.py`) -- so
// nothing here differentiates anything: it is one `Run()` per call.
//
// The MSE loss lives inside the graph, which is why the target picture
// is bound at construction. Taking d(loss)/d(recon) as an input would
// keep the graph loss-agnostic, but no loss of the reconstruction can
// be evaluated before the reconstruction exists, so the caller would
// run the graph once for `recon` and again for the gradient -- doubling
// the cost of every step.

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
    GradSession(const std::string& artifact, const images::ImageArray& target);
    ~GradSession();
    GradSession(GradSession&&) noexcept;
    GradSession& operator=(GradSession&&) noexcept;

    // Suitable for `optimize::run`. The returned function borrows this
    // object, so it must not outlive it.
    optimize::GradFn fn();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace sstvae::codec

#endif
