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

// A part that is already in memory, for a build that *carries* its
// artifacts rather than resolving them to files on disk.
//
// This exists for Android, where the models ship inside the APK. An
// asset is not a filesystem path — there is nothing to hand
// `Ort::Session(env, path, ...)` — so the alternative would be
// extracting ~20 MB to the cache directory on first run and keeping two
// copies of it forever, which is worse in every dimension including the
// one that motivated bundling.
//
// **The bytes are owned, and released as soon as the session is
// built.** ORT parses a model given by pointer into its own structures
// rather than referencing the buffer, so nothing has to keep ~20 MB of
// weights resident for the life of the process — and `session()` pins
// that behaviour with an explicit config entry rather than relying on
// it staying the default.
//
// An owning vector rather than a pointer and a length because the
// natural source is a compressed APK asset, which has to be inflated
// into *somewhere* regardless; handing back a borrowed pointer would
// only move the question of who keeps it alive.
struct ModelBlob {
    std::vector<std::byte> data;
    // Only ever used in error messages; a path is meaningless here, so
    // this is what stands in for one when the checkpoints disagree.
    std::string name;
};

// Consulted *before* the path `Resolver`, and returning nullopt means
// "not bundled, go and find it" rather than an error — which is what
// lets one build carry its models and another fetch them with no
// difference in the code around it.
using BlobResolver =
    std::function<std::optional<ModelBlob>(const std::string& part)>;

class OnnxCodec {
public:
    // `resolver` may be null, in which case `set_resolver` must be
    // called before the first encode or decode.
    explicit OnnxCodec(Resolver resolver = nullptr);
    ~OnnxCodec();
    OnnxCodec(OnnxCodec&&) noexcept;
    OnnxCodec& operator=(OnnxCodec&&) noexcept;

    void set_resolver(Resolver resolver);

    // Either resolver alone is a complete configuration; a codec with a
    // blob resolver and no path resolver is what a fully bundled build
    // has, and it must not be treated as unconfigured.
    void set_blob_resolver(BlobResolver resolver);

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
