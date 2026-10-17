// The encode seam: a picture becomes the stored latents every pass of it
// sends (plan 1.5, stage 1 item 16). The body of qrss_encode.py once the
// image is in memory -- codec.encode(img), then StoredPicture.from_latents.
//
// Header-only and codec-free on purpose. The encoder is a std::function,
// the same pattern as rx::Decoder, so the app and the tests can hand in
// SSTVAE's OnnxCodec (qrss/codec/codec.hpp builds one) or a stub without
// this file knowing about onnxruntime -- anything linking sstvae_qrss can
// include it.

#ifndef SSTVAE_QRSS_CODEC_ENCODE_HPP
#define SSTVAE_QRSS_CODEC_ENCODE_HPP

#include <cstdint>
#include <functional>
#include <stdexcept>
#include <vector>

#include "images/types.hpp"
#include "qrss/picture.hpp"

namespace sstvae::qrss::codec {

// IMG_W x IMG_H planar [0, 1] -> mode C's full flat latent vector in
// canonical order (OnnxCodec::encode's contract: 158,400 values, unit RMS).
using Encoder = std::function<std::vector<double>(const images::ImageArray&)>;

// StoredPicture.from_latents(encode(img), codec_id, mode). Refusals are
// picture.py's (mode, codec ID, a group without energy); a missing
// encoder is std::invalid_argument.
inline picture::StoredPicture encode_picture(const images::ImageArray& img, std::int64_t mode,
                                             std::int64_t codec_id, const Encoder& encode) {
    if (!encode) throw std::invalid_argument("encode_picture: no encoder");
    const std::vector<double> flat = encode(img);
    return picture::StoredPicture::from_latents(flat, codec_id, mode);
}

}  // namespace sstvae::qrss::codec

#endif  // SSTVAE_QRSS_CODEC_ENCODE_HPP
