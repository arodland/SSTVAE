// sstvae-qrss receive's --image, as a hook: qrss_receive.py's render_pass
// needs the codec (onnxruntime), and receive.cpp is built with or without
// it, so the codec half registers itself here from a file compiled only
// with sstvae_qrss_codec (codec/receive_image.cpp). With no renderer
// registered, `sstvae-qrss receive --image` is refused before anything is
// read, with the reason.

#ifndef SSTVAE_APPS_SSTVAE_QRSS_RECEIVE_HPP
#define SSTVAE_APPS_SSTVAE_QRSS_RECEIVE_HPP

#include <optional>
#include <ostream>
#include <string>

#include "qrss/types.hpp"

namespace sstvae::qrss::cli::receive {

// --image, --model and --precision as the script holds them. `image` is
// the path as given; `model` already defaults to $QRSSTVAE_MODEL_DIR
// (render.model_dir) when --model is absent.
struct ImageRequest {
    std::string image;
    std::optional<std::string> model;
    std::string precision;
};

struct ImageRenderer {
    // nullopt when the request can be served; otherwise why not, refused
    // before the receive (the script would fail only after it, or write a
    // format this one does not).
    std::optional<std::string> (*refusal)(const ImageRequest& req) = nullptr;
    // render_pass(p, image, model, precision): writes the lines the script
    // prints for it to `out` (a codec warning, then "image: ...") and
    // returns 0, or ends the run as the script would (an uncaught error or
    // a SystemExit: the message on `err`, and the exit status).
    int (*render)(const types::PassResult& p, const ImageRequest& req, std::ostream& out, std::ostream& err) =
        nullptr;
};

// The registered renderer; both members null when the build has none.
ImageRenderer image_renderer();

struct RegisterImageRenderer {
    explicit RegisterImageRenderer(ImageRenderer r);
};

}  // namespace sstvae::qrss::cli::receive

#endif  // SSTVAE_APPS_SSTVAE_QRSS_RECEIVE_HPP
