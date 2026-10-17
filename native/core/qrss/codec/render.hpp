// The decoder side of QRSSTVAE's codec glue: sstvae/qrss/render.py's load,
// check_codec_id, decoder_codec_id and render (the reference), and the
// placement of one received pass that qrss_receive.py's render_pass makes
// before it renders.
//
// What a picture is made of is render::decoder_planes (qrss/render.hpp,
// bitwise to the reference); this adds only the codec call, SSTVAE's
// OnnxCodec on the same artifact, so the picture is the reference's byte
// for byte wherever CLAUDE.md's codec parity claim holds (the onnxruntime
// version pinned to Python's).
//
// The codec ID check is the reference's: a latent vector decoded by
// another checkpoint's decoder is a picture that is silently wrong, so a
// mismatch is refused (CodecMismatch, render.CodecMismatch's text) unless
// forced, and a decoder without the metadata is a warning.
//
// In sstvae_qrss_codec (it links onnxruntime). Thread-safe: an OpenDecoder's
// decode serialises its calls on its own mutex; nothing else keeps state.

#ifndef SSTVAE_QRSS_CODEC_RENDER_HPP
#define SSTVAE_QRSS_CODEC_RENDER_HPP

#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "checkpoint/checkpoint.hpp"
#include "images/types.hpp"

namespace sstvae::qrss::codec {

// render.CodecMismatch (a SystemExit there): the message.
class CodecMismatch : public std::runtime_error {
public:
    explicit CodecMismatch(const std::string& what) : std::runtime_error(what) {}
};

// render.check_codec_id(codec, codec_id, force), given the loaded decoder's
// codec ID `have` (decoder_codec_id: nullopt when it carries none): nullopt
// when codec_id is nullopt or matches; the warning when `have` is nullopt;
// otherwise CodecMismatch, or with `force` the warning saying it decodes
// anyway. The texts are the reference's (hex codec IDs, "--any-codec").
std::optional<std::string> check_codec_id(std::optional<std::int64_t> have, std::optional<std::int64_t> codec_id,
                                          bool force = false);

// The decoder half of codec.load_codec(model, precision), loaded.
struct OpenDecoder {
    std::string onnx_path;                  // resolve_onnx("decoder", model, precision), as pathlib prints it
    std::optional<std::int64_t> codec_id;   // render.decoder_codec_id
    // codec.decode(latents, weights): float32[158400] each, canonical order.
    std::function<images::Picture(std::span<const float> latents, std::span<const float> weights)> decode;
};

// Locate and load the decoder (its session built now, as decoder_codec_id
// does in the reference). Refusals: std::invalid_argument "precision applies
// to the ONNX backend only" for a ".pt" model (load_codec's torch backend,
// which refuses any precision), ModelError / CheckpointError (resolve_onnx's
// SystemExit texts), OrtError for an artifact onnxruntime cannot load.
OpenDecoder open_decoder(const std::optional<std::string>& model,
                         std::string_view precision = "fp32", const checkpoint::Fetcher& fetcher = nullptr);

// render.load(precision, model, codec_id, force): open_decoder, then
// check_codec_id against it; the warning (if any) beside the decoder, for
// the caller to print where the reference's `warn` would.
struct LoadedDecoder {
    OpenDecoder decoder;
    std::optional<std::string> warning;
};
LoadedDecoder load(std::string_view precision = "fp32", const std::optional<std::string>& model = std::nullopt,
                   std::optional<std::int64_t> codec_id = std::nullopt, bool force = false,
                   const checkpoint::Fetcher& fetcher = nullptr);

// render.render(codec, S, W, mode): decoder_planes(S, W, mode), decoded.
// S, W float64[3 * 52800] (render::decoder_planes' refusals).
images::Picture render(const OpenDecoder& decoder, std::span<const double> S, std::span<const double> W,
                       std::int64_t mode);

// qrss_receive.py render_pass's planes for one pass of group `segment`:
// S[segment, air_to_canonical(segment)] = w * z and W[...] = w, in float64
// from the pass's float32 z and w (SENT values each, air order), zero
// elsewhere. std::invalid_argument when z and w are not SENT long or the
// segment is not 0..2.
std::pair<std::vector<double>, std::vector<double>> pass_planes(std::span<const float> z, std::span<const float> w,
                                                                std::int64_t segment);

}  // namespace sstvae::qrss::codec

#endif  // SSTVAE_QRSS_CODEC_RENDER_HPP
