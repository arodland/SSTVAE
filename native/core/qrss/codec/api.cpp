// The codec half of the in-process QRSSTVAE transmit API (api.hpp).

#include "qrss/codec/api.hpp"

#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "qrss/codec/codec.hpp"

namespace sstvae::qrss::api {

OpenEncoder open_encoder(const std::optional<std::string>& model, std::string_view precision,
                         std::optional<std::int64_t> codec_id, const checkpoint::Fetcher& fetcher) {
    if (codec_id) picture::check_codec_id(*codec_id);
    OpenEncoder out;
    out.onnx_path = codec::resolve_onnx("encoder", model, precision, fetcher);
    if (codec_id) {
        out.codec_id = *codec_id;
    } else {
        const std::optional<std::int64_t> found = codec::codec_id_from_onnx(out.onnx_path, precision);
        if (!found)
            throw std::invalid_argument(out.onnx_path +
                                        " carries no sstvae.source_sha256 metadata; give its codec ID so "
                                        "receivers know which decoder to use");
        out.codec_id = *found;
    }
    // The resolved artifact itself, so the codec does not resolve (or
    // fetch) again; OnnxCodec is lazy and not thread-safe, hence the lock.
    auto inner = std::make_shared<Encoder>(codec::onnx_encoder(out.onnx_path, precision, fetcher));
    auto lock = std::make_shared<std::mutex>();
    out.encode = [inner, lock](const images::ImageArray& img) {
        std::lock_guard<std::mutex> guard(*lock);
        return (*inner)(img);
    };
    return out;
}

picture::StoredPicture encode_file(const std::string& image_path, std::int64_t mode, const OpenEncoder& encoder) {
    picture::check_mode(mode);
    return codec::encode_picture(image_path, mode, encoder.codec_id, encoder.encode);
}

}  // namespace sstvae::qrss::api
