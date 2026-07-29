// The codec: pictures in, latents out, and back.
//
// onnxruntime, CPU, no torch -- same as the Python side, and for the
// same reasons (docs/onnx.md). The published fp16 artifacts are the
// default; `docs/onnx.md` measures fp16 as identical to fp32 end to end.
//
// Parity is stronger here than anywhere else in the port, because both
// implementations call the *same* library on the *same* file rather
// than reimplementing an algorithm: the encoder is bit-identical to
// Python's and the decoder byte-identical on the published artifacts.
// Two things are required for that and neither is free (see the notes
// on `quantize` below and on the version pin in
// native/cmake/onnxruntime.cmake).

#ifndef SSTVAE_CODEC_CODEC_HPP
#define SSTVAE_CODEC_CODEC_HPP

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "config.hpp"
#include "images/types.hpp"
#include "latents/latents.hpp"

namespace sstvae::codec {

using images::ImageArray;
using images::IMG_H;
using images::IMG_W;
using images::Picture;

// Both live in core/latents/ so the rx engine can pad a short mode's
// latents without linking onnxruntime; re-exported here because
// `codec::pad_to_full` is how every caller spells it.
using latents::N_LATENTS;
using latents::pad_to_full;

// Which artifact to load for a part ("encoder" / "decoder").
//
// A seam rather than a hardcoded path so the Hub fetch, an explicit
// --model, and a test fixture all plug in the same way. `checkpoint.hpp`
// supplies the default.
using Resolver = std::function<std::string(const std::string& part)>;

class OnnxCodec {
public:
    // `resolver` may be null, in which case `set_resolver` must be
    // called before the first encode or decode.
    explicit OnnxCodec(Resolver resolver = nullptr);
    ~OnnxCodec();
    OnnxCodec(OnnxCodec&&) noexcept;
    OnnxCodec& operator=(OnnxCodec&&) noexcept;

    void set_resolver(Resolver resolver);

    // Picture -> mode C's full-length latent vector, unit RMS.
    // Callers truncate to their mode; the normalization happens inside
    // the graph, and must not be repeated anywhere else.
    std::vector<double> encode(const ImageArray& image);

    // Full-length latent/weight vectors -> picture. Erased latents are
    // zeroed rather than merely down-weighted, matching Python.
    Picture decode(const std::vector<double>& latents,
                   const std::vector<double>& weights);

    // Load a part now rather than on first use. Only useful for
    // surfacing a missing-artifact error early -- the parts are
    // otherwise deliberately lazy and independent, so a receive-only
    // station never touches the encoder.
    void preload(const std::string& part);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace sstvae::codec

#endif
