#include "qrss/codec/render.hpp"

#include <cstddef>
#include <memory>
#include <mutex>

#include "codec/codec.hpp"
#include "qrss/codec/codec.hpp"
#include "qrss/picture.hpp"
#include "qrss/render.hpp"

namespace sstvae::qrss::codec {

namespace {

// f"{v:04x}" of a codec ID (16 bits, never negative).
std::string hex4(std::int64_t v) {
    std::string digits;
    auto u = static_cast<std::uint64_t>(v < 0 ? -v : v);
    do {
        digits.insert(digits.begin(), "0123456789abcdef"[u % 16]);
        u /= 16;
    } while (u);
    const std::size_t width = v < 0 ? 3 : 4;   // the sign counts toward the width
    if (digits.size() < width) digits.insert(0, width - digits.size(), '0');
    return (v < 0 ? "-" : "") + digits;
}

}  // namespace

std::optional<std::string> check_codec_id(std::optional<std::int64_t> have, std::optional<std::int64_t> codec_id,
                                          bool force) {
    if (!codec_id) return std::nullopt;
    if (!have)
        return "warning: the decoder carries no codec ID; the picture's codec " + hex4(*codec_id) +
               " cannot be checked against it";
    if (*have == *codec_id) return std::nullopt;
    const std::string msg = "the picture was sent with codec " + hex4(*codec_id) + " but the decoder loaded is codec " +
                            hex4(*have) +
                            ": it would decode to a silently wrong picture. Load the matching model with --model";
    if (!force) throw CodecMismatch(msg + " (or pass --any-codec to decode anyway).");
    return "warning: " + msg + "; decoding anyway (--any-codec).";
}

OpenDecoder open_decoder(const std::optional<std::string>& model, std::string_view precision,
                         const checkpoint::Fetcher& fetcher) {
    // codec.load_codec: a .pt is the torch backend, which refuses a precision.
    if (model && pathlib_suffix(*model) == ".pt")
        throw std::invalid_argument("precision applies to the ONNX backend only");
    OpenDecoder out;
    out.onnx_path = resolve_onnx("decoder", model, precision, fetcher);
    // decoder_codec_id: the session's sstvae.source_sha256, int(sha[:4], 16).
    if (const std::optional<std::string> sha = source_sha256(out.onnx_path)) out.codec_id = codec_id_of_sha(*sha);
    // The resolved artifact itself, so the codec does not resolve (or fetch)
    // again; OnnxCodec is lazy and not thread-safe, hence the lock.
    auto codec = std::make_shared<::sstvae::codec::OnnxCodec>([path = out.onnx_path](const std::string& part) {
        if (part != "decoder") throw std::invalid_argument("this codec holds the decoder only, not the " + part);
        return path;
    });
    codec->preload("decoder");
    auto lock = std::make_shared<std::mutex>();
    out.decode = [codec, lock](std::span<const float> latents, std::span<const float> weights) {
        const std::vector<double> lat(latents.begin(), latents.end()), wt(weights.begin(), weights.end());
        std::lock_guard<std::mutex> guard(*lock);
        return codec->decode(lat, wt);
    };
    return out;
}

LoadedDecoder load(std::string_view precision, const std::optional<std::string>& model,
                   std::optional<std::int64_t> codec_id, bool force, const checkpoint::Fetcher& fetcher) {
    LoadedDecoder out;
    out.decoder = open_decoder(model, precision, fetcher);
    out.warning = check_codec_id(out.decoder.codec_id, codec_id, force);
    return out;
}

images::Picture render(const OpenDecoder& decoder, std::span<const double> S, std::span<const double> W,
                       std::int64_t mode) {
    const auto [lat, wt] = qrss::render::decoder_planes(S, W, mode);
    return decoder.decode(lat, wt);
}

std::pair<std::vector<double>, std::vector<double>> pass_planes(std::span<const float> z, std::span<const float> w,
                                                                std::int64_t segment) {
    if (z.size() != static_cast<std::size_t>(picture::SENT) || w.size() != z.size())
        throw std::invalid_argument("a pass of " + std::to_string(z.size()) + " latents and " +
                                    std::to_string(w.size()) + " weights is not a picture segment (" +
                                    std::to_string(picture::SENT) + ")");
    const std::span<const std::uint16_t> idx = picture::air_to_canonical(segment);
    const std::size_t n = static_cast<std::size_t>(picture::N_GROUPS) * picture::GROUP_LATENTS;
    std::vector<double> S(n, 0.0), W(n, 0.0);
    const std::size_t base = static_cast<std::size_t>(segment) * picture::GROUP_LATENTS;
    for (std::size_t k = 0; k < idx.size(); ++k) {
        const double wk = w[k];
        // float32 * float32 in float64 is exact: S is w z to the bit.
        S[base + idx[k]] = wk * static_cast<double>(z[k]);
        W[base + idx[k]] = wk;
    }
    return {std::move(S), std::move(W)};
}

}  // namespace sstvae::qrss::codec
