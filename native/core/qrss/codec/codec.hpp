// QRSSTVAE's codec glue (plan 1.5, stage 1 item 16): where a stored picture
// meets SSTVAE's codec on the transmit side.
//
// - The codec ID. A .qrsp records which checkpoint its latents came from,
//   as the first two bytes of the model artifact's `sstvae.source_sha256`
//   metadata (decision D11); v5 is 0xD1D8. Port of
//   picture.codec_id_from_onnx (sstvae/qrss/picture.py, the reference).
// - The encode of an image file into a StoredPicture, exactly as
//   qrss_encode.py does it: images.load_image (open, fit, to [0, 1]), the
//   codec's encode, StoredPicture.from_latents.
//
// A library of its own, sstvae_qrss_codec, built only when SSTVAE's codec
// is (SSTVAE_BUILD_CODEC): it links onnxruntime, which reads the metadata,
// and everything else in the QRSS port must build and test without it.
// encode.hpp is the codec-free seam this builds on.
//
// The model is located as sstvae/checkpoint.py's resolve_onnx does,
// with that function's refusal texts (ModelError), so `sstvae-qrss encode`
// refuses a --model the way qrss_encode.py does. With no --model the
// published artifact comes from SSTVAE's native cache (checkpoint.hpp):
// the native app keeps its own cache rather than huggingface_hub's, by
// design, so that one path is the native's and not the reference's.

#ifndef SSTVAE_QRSS_CODEC_CODEC_HPP
#define SSTVAE_QRSS_CODEC_CODEC_HPP

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "checkpoint/checkpoint.hpp"
#include "qrss/codec/encode.hpp"
#include "qrss/picture.hpp"
#include "qrss/pypath.hpp"

namespace sstvae::qrss::codec {

// --- Python's text conventions these refusals need --------------------------

// str(pathlib.PurePosixPath(text)) and its .suffix: qrss/pypath.hpp's, the
// one copy the CLI shares, named here as they always were.
using pypath::pathlib_str;
using pypath::pathlib_suffix;

// --- locating the model ---------------------------------------------------------

// checkpoint.resolve_onnx's refusals (SystemExit in the reference), with its
// texts; a CheckpointError, so a caller catching SSTVAE's sees these too.
class ModelError : public checkpoint::CheckpointError {
public:
    explicit ModelError(const std::string& what) : checkpoint::CheckpointError(what) {}
};

// sstvae/checkpoint.py's resolve_onnx(part, path, precision): a directory
// holds exactly one `*-{part}-{precision}.onnx`; a `.onnx` names that part,
// or the other part, whose sibling is derived from it; anything else is
// refused. Paths come back as pathlib prints them. With no path, SSTVAE's
// native resolver finds the published artifact (cache, then `fetcher` or
// the default one) and its CheckpointError texts are its own.
std::string resolve_onnx(std::string_view part, const std::optional<std::string>& path,
                         std::string_view precision = checkpoint::DEFAULT_PRECISION,
                         const checkpoint::Fetcher& fetcher = nullptr);

// --- the codec ID ------------------------------------------------------------------

// codec_id_from_onnx's FileNotFoundError: a directory with neither part.
class NoModelFile : public std::runtime_error {
public:
    explicit NoModelFile(const std::string& what) : std::runtime_error(what) {}
};

// onnxruntime refused the artifact. what() is the message as the Python
// binding words it ("[ONNXRuntimeError] : 3 : NO_SUCHFILE : Load model
// from x failed:..."), `python_class` the exception class it raises there
// ("NoSuchFile", in onnxruntime.capi.onnxruntime_pybind11_state).
class OrtError : public std::runtime_error {
public:
    OrtError(int code, std::string python_class, const std::string& what)
        : std::runtime_error(what), code_(code), python_class_(std::move(python_class)) {}
    int code() const noexcept { return code_; }
    const std::string& python_class() const noexcept { return python_class_; }

private:
    int code_;
    std::string python_class_;
};

// The artifact's `sstvae.source_sha256` metadata, nullopt when absent. Opens
// an onnxruntime session on `onnx_path` (as the reference does), so a file
// onnxruntime cannot load is OrtError.
std::optional<std::string> source_sha256(const std::string& onnx_path);

// The codec ID a source hash gives: int(sha[:4], 16), nullopt when the hash
// is empty, shorter than 4 characters or not a hex literal there (the
// reference's None). Characters are code points, as Python counts them.
std::optional<std::int64_t> codec_id_of_sha(std::string_view sha);

// picture.codec_id_from_onnx(path_or_dir, precision): the first two bytes
// of the artifact's source hash. A directory resolves the encoder, then the
// decoder (both carry the same hash); neither is NoModelFile. nullopt when
// the metadata is absent or unreadable as hex.
std::optional<std::int64_t> codec_id_from_onnx(const std::string& path_or_dir,
                                               std::string_view precision = checkpoint::DEFAULT_PRECISION);

// --- the encode ----------------------------------------------------------------------

// SSTVAE's OnnxCodec as an Encoder, its encoder part located by
// resolve_onnx(path, precision) above when first used (lazily, as the
// reference's load_codec). The codec lives as long as the returned function.
Encoder onnx_encoder(const std::optional<std::string>& model,
                     std::string_view precision = checkpoint::DEFAULT_PRECISION,
                     const checkpoint::Fetcher& fetcher = nullptr);

// images.load_image(path): open (EXIF orientation applied), fit to IMG_W x
// IMG_H by cover and centre crop, to planar [0, 1]. std::runtime_error from
// the image layer on a file it cannot read.
images::ImageArray load_image(const std::string& path);

// qrss_encode.py's work after the arguments: load_image(image_path),
// encode, StoredPicture.from_latents(flat, codec_id, mode). A 640x480 PNG
// gives the reference's picture bit for bit (the resize of any other size
// is stb's, not PIL's LANCZOS -- an accepted difference above the modem).
picture::StoredPicture encode_picture(const std::string& image_path, std::int64_t mode, std::int64_t codec_id,
                                      const Encoder& encode);

}  // namespace sstvae::qrss::codec

#endif  // SSTVAE_QRSS_CODEC_CODEC_HPP
