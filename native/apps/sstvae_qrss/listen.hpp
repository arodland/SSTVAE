// sstvae-qrss listen's pictures, as a hook: the live listener draws its
// tiles with the codec (onnxruntime), and listen.cpp is built with or
// without it, so the codec half registers the renderer's factory here from
// a file compiled only with sstvae_qrss_codec (codec/listen_render.cpp).
// With none registered, `listen` runs as with --no-pictures and says so
// once (listen_api.hpp's NO_PICTURES_NOTICE).

#ifndef SSTVAE_APPS_SSTVAE_QRSS_LISTEN_HPP
#define SSTVAE_APPS_SSTVAE_QRSS_LISTEN_HPP

#include "qrss/listen_api.hpp"

namespace sstvae::qrss::cli::listen_render {

// The registered factory; empty when the build has none.
qrss::listen::RendererFactory factory();

struct Register {
    explicit Register(qrss::listen::RendererFactory f);
};

}  // namespace sstvae::qrss::cli::listen_render

#endif  // SSTVAE_APPS_SSTVAE_QRSS_LISTEN_HPP
