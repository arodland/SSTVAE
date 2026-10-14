// The codec half of qrss_decode.py: render.render of the planes it built
// (decode::planes_from_pass) and render.render_accumulator of a store's
// accumulator, through the decoder render::load opened (codec/render.hpp).
//
// The picture is render::decoder_planes (bitwise the reference's) through
// SSTVAE's decoder on the same artifact, so it is the reference's pixel for
// pixel wherever CLAUDE.md's codec parity claim holds.
//
// In sstvae_qrss_codec (onnxruntime). Thread-safe as OpenDecoder is.

#ifndef SSTVAE_QRSS_CODEC_DECODE_HPP
#define SSTVAE_QRSS_CODEC_DECODE_HPP

#include "images/types.hpp"
#include "qrss/codec/render.hpp"
#include "qrss/decode.hpp"
#include "qrss/store.hpp"

namespace sstvae::qrss::codec {

// render.render(codec, S, W, mode) of decode::Planes.
images::Picture render_planes(const OpenDecoder& decoder, const decode::Planes& planes);

// render.render_accumulator(codec, acc): render(codec, acc.S, acc.W, acc.mode),
// the float32 planes widened to float64.
images::Picture render_accumulator(const OpenDecoder& decoder, const store::Accumulator& acc);

}  // namespace sstvae::qrss::codec

#endif  // SSTVAE_QRSS_CODEC_DECODE_HPP
