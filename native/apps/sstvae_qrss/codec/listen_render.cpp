// sstvae-qrss listen's pictures: the codec half of the listen subcommand
// (listen.hpp's hook), codec::listener_renderer -- live.py's _codec and
// _render: the decoder loaded at the first picture with --precision and
// --model (else $QRSSTVAE_MODEL_DIR, else the published artifacts). Built
// only with sstvae_qrss_codec, as codec/decode_image.cpp is.

#include <memory>

#include "../listen.hpp"
#include "qrss/codec/listener_render.hpp"

namespace sstvae::qrss::cli {
namespace {

const listen_render::Register reg([](const live::LiveConfig& cfg) -> std::shared_ptr<live::Renderer> {
    return codec::listener_renderer(cfg);
});

}  // namespace
}  // namespace sstvae::qrss::cli
