// sstvae-qrss receive --image: qrss_receive.py's render_pass, the codec half
// of the receive subcommand (receive.hpp's hook). Compiled only with the
// codec glue (sstvae_qrss_codec): it is outside apps/sstvae_qrss/*.cpp's
// glob, so it is built only where native/CMakeLists.txt adds it to
// sstvae_qrss_cli inside `if(TARGET sstvae_qrss_codec)`:
//     target_sources(sstvae_qrss_cli PRIVATE
//                    ${CMAKE_CURRENT_SOURCE_DIR}/apps/sstvae_qrss/codec/receive_image.cpp)
// Without that line the subcommand refuses --image (receive.cpp).
//
// render_pass(p, image, model, precision), step by step:
//   no header                      "image: no header decoded, cannot place the latents"
//   len(p.z) != SENT               "image: a <frame> frame holds <n> latents, not a picture segment"
//   S, W: the pass at its segment's canonical places (codec::pass_planes)
//   render.load(precision, model, codec_id=h.codec_id, warn=print)
//         codec mismatch           "image: not written: <CodecMismatch's text>"
//         no codec ID on the decoder: the warning, printed
//         resolve_onnx's refusals  SystemExit (exit 1); onnxruntime's: uncaught
//   render.render(codec, S, W, h.mode).save(image)
//   "image: <image>"
//
// Pillow's save picks the format by the file's extension; the native image
// layer writes PNG only, so any other extension is refused before the
// receive (receive.hpp's `refusal`) rather than written as PNG under a name
// that says otherwise. The PNG's pixels are the reference's (the codec's
// parity claim); its bytes are stb's encoder's, not Pillow's.

#include <cerrno>
#include <cstdio>
#include <cstdint>
#include <exception>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <string>

#include "../common.hpp"
#include "../receive.hpp"
#include "checkpoint/checkpoint.hpp"
#include "images/images.hpp"
#include "qrss/codec/codec.hpp"
#include "qrss/codec/render.hpp"
#include "qrss/picture.hpp"
#include "qrss/types.hpp"
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

std::optional<std::string> refusal(const receive::ImageRequest& req) {
    const std::string path = qc::pathlib_str(req.image);
    const std::string ext = extension(path);
    if (ext == ".png" || ext == ".apng") return std::nullopt;
    return "--image " + path + ": sstvae-qrss writes PNG only (a .png name); qrss_receive.py --image writes "
           "whatever format Pillow names by the extension";
}

int render_pass(const types::PassResult& p, const receive::ImageRequest& req, std::ostream& out,
                std::ostream& err) {
    if (!p.header) {
        out << "image: no header decoded, cannot place the latents\n";
        return 0;
    }
    const header::HeaderFields& h = *p.header;
    if (p.z.size() != static_cast<std::size_t>(picture::SENT)) {
        out << "image: a " << p.frame << " frame holds " << p.z.size() << " latents, not a picture segment\n";
        return 0;
    }
    try {
        const auto [S, W] = qc::pass_planes(p.z, p.w, h.segment);
        qc::LoadedDecoder loaded;
        try {
            loaded = qc::load(req.precision, req.model, h.codec_id);
        } catch (const qc::CodecMismatch& e) {
            out << "image: not written: " << e.what() << "\n";
            return 0;
        }
        if (loaded.warning) out << *loaded.warning << "\n";
        const images::Picture pic = qc::render(loaded.decoder, S, W, h.mode);
        // Image.save(Path(image)): the format from the name, then
        // open(name, "w+b") -- an OSError with errno -- then the encoder.
        const std::string path = qc::pathlib_str(req.image);
        errno = 0;
        if (!util::open_file(path, "wb")) return uncaught(err, OsError(errno ? errno : EIO, path));
        images::save_png(pic, path);
        out << "image: " << path << "\n";
        return 0;
    } catch (const checkpoint::CheckpointError& e) {
        // checkpoint.resolve_onnx's SystemExit.
        return system_exit(err, e.what());
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

const receive::RegisterImageRenderer reg(receive::ImageRenderer{&refusal, &render_pass});

}  // namespace
}  // namespace sstvae::qrss::cli
