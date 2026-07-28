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

namespace sstvae::codec {

// Target geometry. `sstvae/images.py` states these as literals; here
// they are tied to the latent grid, which is what actually fixes them
// -- the decoder upsamples by 16 in each axis, so the picture size is
// not an independent choice and a static_assert is more honest than a
// second copy of the numbers.
inline constexpr int UPSAMPLE = 16;
inline constexpr int IMG_W = config::LATENT_W * UPSAMPLE;
inline constexpr int IMG_H = config::LATENT_H * UPSAMPLE;
static_assert(IMG_W == 640 && IMG_H == 480, "picture geometry moved");

inline constexpr int N_LATENTS =
    config::LATENT_CHANNELS * config::LATENT_H * config::LATENT_W;

// 8-bit RGB, interleaved, row-major -- the layout PIL hands back, so a
// picture crosses the binding without a transpose.
struct Picture {
    int width = 0;
    int height = 0;
    std::vector<std::uint8_t> rgb;  // height * width * 3

    Picture() = default;
    Picture(int w, int h) : width(w), height(h), rgb(static_cast<std::size_t>(w) * h * 3) {}
    bool empty() const { return rgb.empty(); }
};

// Planar float in [0,1], (3, H, W) -- what the encoder graph takes, and
// what `images::to_array` produces.
struct ImageArray {
    int width = 0;
    int height = 0;
    std::vector<float> chw;  // 3 * height * width
};

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

// Extend a mode A/B latent (or weight) vector to mode C's length.
// The modes are nested, so a shorter one is a full-length vector whose
// tail never arrived -- which is exactly what weight 0 means.
std::vector<double> pad_to_full(const std::vector<double>& vec, double fill = 0.0);

}  // namespace sstvae::codec

#endif
