// sstvae-qrss decode's picture: qrss_decode.py's last step, the codec half
// of the decode subcommand (decode.hpp's hook). Compiled only with the
// codec glue (sstvae_qrss_codec): it is outside apps/sstvae_qrss/*.cpp's
// glob, so it is built only where native/CMakeLists.txt adds
// apps/sstvae_qrss/codec/*.cpp to sstvae_qrss_cli inside
// `if(TARGET sstvae_qrss_codec)`. Without it the subcommand refuses every
// picture at this step (decode.cpp).
//
// The step:
//   render.load(precision, model, codec_id=codec_id, force=any_codec, warn=print)
//         codec mismatch without --any-codec   SystemExit (CodecMismatch's text), exit 1
//         a warning (no codec ID on the decoder, or --any-codec over a mismatch): printed
//         resolve_onnx's refusals             SystemExit (exit 1); onnxruntime's: uncaught
//   render.render(codec, S, W, mode)           render::decoder_planes' ValueErrors
//   .save(output)                              open(name, "w+b"): OSError with errno
//   "wrote <output>"
//
// Pillow's save picks the format by the file's extension; the native image
// layer writes PNG only, so any other extension is refused before the codec
// is loaded (decode.hpp's `refusal`) rather than written as PNG under a
// name that says otherwise. The PNG's pixels are the reference's (the
// codec's parity claim); its bytes are stb's encoder's, not Pillow's.

#include <cerrno>
#include <cstdint>
#include <exception>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <string>

#include "../common.hpp"
#include "../decode.hpp"
#include "checkpoint/checkpoint.hpp"
#include "images/images.hpp"
#include "qrss/codec/codec.hpp"
#include "qrss/codec/decode.hpp"
#include "qrss/codec/render.hpp"
#include "qrss/decode.hpp"
#include "qrss/png.hpp"
#include "qrss/rawfile.hpp"
#include "util/fileio.hpp"

namespace sstvae::qrss::cli {
namespace {

namespace qc = sstvae::qrss::codec;

// os.path.splitext(path)[1].lower(): the extension Pillow looks the format
// up by (a name of leading dots has none).
std::string extension(const std::string& path) {
    const std::size_t slash = path.rfind('/');
    const std::size_t start = slash == std::string::npos ? 0 : slash + 1;
    const std::size_t dot = path.rfind('.');
    if (dot == std::string::npos || dot < start) return {};
    std::size_t k = start;
    while (k < dot && path[k] == '.') ++k;
    if (k == dot) return {};
    std::string ext = path.substr(dot);
    for (char& ch : ext)
        if (ch >= 'A' && ch <= 'Z') ch = static_cast<char>(ch - 'A' + 'a');
    return ext;
}

std::optional<std::string> refusal(const decode_image::PictureRequest& req) {
    const std::string path = qc::pathlib_str(req.output);
    const std::string ext = extension(path);
    if (ext == ".png" || ext == ".apng") return std::nullopt;
    return path + ": sstvae-qrss decode writes PNG only (a .png name); qrss_decode.py writes whatever format "
                  "Pillow names by the extension";
}

int render_picture(const qrss::decode::Planes& planes, std::int64_t codec_id,
                   const decode_image::PictureRequest& req, std::ostream& out, std::ostream& err) {
    try {
        qc::LoadedDecoder loaded;
        try {
            loaded = qc::load(req.precision, req.model, codec_id, req.any_codec);
        } catch (const qc::CodecMismatch& e) {
            return system_exit(err, e.what());   // render.CodecMismatch is a SystemExit
        }
        if (loaded.warning) out << *loaded.warning << "\n";
        const images::Picture pic = qc::render_planes(loaded.decoder, planes);
        // Image.save(Path(output)): the format from the name, then
        // open(name, "w+b") -- an OSError with errno -- then the encoder.
        const std::string path = qc::pathlib_str(req.output);
        errno = 0;
        if (!util::open_file(path, "wb")) return uncaught(err, OsError(errno ? errno : EIO, path));
        // The bytes through util/fileio.hpp (images::save_png's narrow fopen
        // cannot open a name outside the ANSI code page on Windows): the same
        // PNG, by way of a temporary sibling moved into place.
        try {
            png::write_png(pic, path);
        } catch (const rawfile::FileError& e) {
            return uncaught(err, OsError(e.code().value(), path));
        }
        out << "wrote " << path << "\n";
        return 0;
    } catch (const checkpoint::CheckpointError& e) {
        return system_exit(err, e.what());   // checkpoint.resolve_onnx's SystemExit
    } catch (const qc::OrtError& e) {
        return uncaught(err, "onnxruntime.capi.onnxruntime_pybind11_state." + e.python_class(), e.what());
    } catch (const std::invalid_argument& e) {
        return uncaught(err, "ValueError", e.what());
    } catch (const PyError& e) {
        return uncaught(err, e);
    } catch (const std::exception& e) {
        return uncaught(err, "RuntimeError", e.what());
    }
}

const decode_image::RegisterPictureRenderer reg(decode_image::PictureRenderer{&refusal, &render_picture});

}  // namespace
}  // namespace sstvae::qrss::cli
