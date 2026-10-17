// sstvae-qrss decode's picture output, as a hook: qrss_decode.py's last
// step (render.load, render.render, Image.save) needs the codec
// (onnxruntime), and decode.cpp is built with or without it, so the codec
// half registers itself here from a file compiled only with
// sstvae_qrss_codec (codec/decode_image.cpp). With no renderer registered,
// a decode is refused at that step, after everything the script prints
// before it (the accumulator's lines, a pass's line, every refusal on the
// way), with the reason.

#ifndef SSTVAE_APPS_SSTVAE_QRSS_DECODE_HPP
#define SSTVAE_APPS_SSTVAE_QRSS_DECODE_HPP

#include <cstdint>
#include <optional>
#include <ostream>
#include <string>

#include "qrss/decode.hpp"

namespace sstvae::qrss::cli::decode_image {

// OUT.png, --model, --precision and --any-codec as the script holds them.
// `output` is the path as given; `model` already defaults to
// $QRSSTVAE_MODEL_DIR (render.model_dir) when --model is absent.
struct PictureRequest {
    std::string output;
    std::optional<std::string> model;
    std::string precision;
    bool any_codec = false;
};

struct PictureRenderer {
    // nullopt when the request can be served; otherwise why not (a format
    // this build does not write), refused before the codec is loaded.
    std::optional<std::string> (*refusal)(const PictureRequest& req) = nullptr;
    // render.load(precision, model, codec_id=codec_id, force=any_codec,
    // warn=print), render.render(codec, S, W, mode).save(output),
    // print(f"wrote {output}"): the lines on `out`, 0; or the script's end
    // (a SystemExit or an uncaught error on `err`) and its status.
    int (*render)(const qrss::decode::Planes& planes, std::int64_t codec_id, const PictureRequest& req,
                  std::ostream& out, std::ostream& err) = nullptr;
};

// The registered renderer; both members null when the build has none.
PictureRenderer picture_renderer();

struct RegisterPictureRenderer {
    explicit RegisterPictureRenderer(PictureRenderer r);
};

}  // namespace sstvae::qrss::cli::decode_image

#endif  // SSTVAE_APPS_SSTVAE_QRSS_DECODE_HPP
