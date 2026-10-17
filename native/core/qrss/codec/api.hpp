// The codec half of the in-process QRSSTVAE transmit API (qrss/api.hpp):
// an encoder from a model, with the codec ID a stored picture records.
// In sstvae_qrss_codec, since it links onnxruntime.
//
// qrss_encode.py's model handling: resolve_onnx("encoder", model,
// precision), the codec ID from the artifact's sstvae.source_sha256
// metadata (or the one given, its --codec-id), then SSTVAE's OnnxCodec.
// With no model the published artifact comes from SSTVAE's native cache
// (fetched with `fetcher`, or the default one, on first use).
//
// A send then needs only api::encode_picture(picture, mode,
// encoder.codec_id, encoder.encode).

#ifndef SSTVAE_QRSS_CODEC_API_HPP
#define SSTVAE_QRSS_CODEC_API_HPP

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include "checkpoint/checkpoint.hpp"
#include "qrss/api.hpp"
#include "qrss/picture.hpp"

namespace sstvae::qrss::api {

struct OpenEncoder {
    std::string onnx_path;     // the encoder artifact, as resolve_onnx prints it
    std::int64_t codec_id = 0; // what the stored pictures it makes record
    // OnnxCodec's encode behind a mutex: safe to call from several threads,
    // one encode at a time. The session is built on the first call.
    Encoder encode;
};

// Refusals: codec::ModelError (resolve_onnx's texts; a CheckpointError) for
// a model that is not there, codec::OrtError for an artifact onnxruntime
// cannot load, codec::NoModelFile for a directory with neither part,
// std::invalid_argument for a codec_id outside 16 bits or a model without
// the metadata and no codec_id ("<path> carries no sstvae.source_sha256
// metadata; give its codec ID so receivers know which decoder to use").
OpenEncoder open_encoder(const std::optional<std::string>& model,
                         std::string_view precision = checkpoint::DEFAULT_PRECISION,
                         std::optional<std::int64_t> codec_id = std::nullopt,
                         const checkpoint::Fetcher& fetcher = nullptr);

// qrss_encode.py IMAGE OUT --mode M with this encoder: the image file read
// as images.load_image reads it (EXIF orientation, fit, [0, 1]), then
// encoded. std::runtime_error from the image layer for a file it cannot read.
picture::StoredPicture encode_file(const std::string& image_path, std::int64_t mode, const OpenEncoder& encoder);

}  // namespace sstvae::qrss::api

#endif  // SSTVAE_QRSS_CODEC_API_HPP
