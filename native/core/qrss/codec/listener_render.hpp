// The live listener's pictures drawn by the codec: live.py's _codec (the
// lazy render.load) and the render.check_codec_id / render.render calls of
// its _render, behind live::Renderer (qrss/listener.hpp), which is what the
// listener asks for a picture.
//
// The decoder is loaded the first time a picture is drawn, with the
// listener's precision and model (no model: $QRSSTVAE_MODEL_DIR, else the
// published artifacts, render.model_dir()), and checked against the first
// picture's codec ID then, as render.load(precision, model, codec_id=...,
// warn=...) does. A load that fails is remembered and never tried again:
// its text is what the listener shows as codec_error (str(e), or the
// exception's type name when that is empty).
//
// In sstvae_qrss_codec (it links onnxruntime). Thread-safe: the listener's
// two workers may draw at once; the decoder serialises its own calls.

#ifndef SSTVAE_QRSS_CODEC_LISTENER_RENDER_HPP
#define SSTVAE_QRSS_CODEC_LISTENER_RENDER_HPP

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>

#include "checkpoint/checkpoint.hpp"
#include "images/types.hpp"
#include "qrss/codec/render.hpp"
#include "qrss/listener.hpp"

namespace sstvae::qrss::codec {

class ListenerRenderer : public live::Renderer {
public:
    // precision and model as LiveConfig's; fetcher: how a missing artifact
    // is downloaded (null: checkpoint's default).
    explicit ListenerRenderer(std::string precision = "fp32", std::optional<std::string> model = std::nullopt,
                              checkpoint::Fetcher fetcher = nullptr);

    // live::Renderer
    std::optional<std::string> load(std::optional<std::int64_t> codec_id, const Warn& warn) override;
    std::optional<std::string> check(std::optional<std::int64_t> codec_id) override;
    images::Picture render(std::span<const double> S, std::span<const double> W, std::int64_t mode,
                           std::optional<std::int64_t> codec_id) override;

    // Loaded, and the loaded decoder's codec ID (nullopt: none, or not loaded).
    bool loaded() const;
    std::optional<std::int64_t> decoder_codec_id() const;
    // The load's error, once it failed.
    std::optional<std::string> error() const;
    // The model asked for: the one given, else $QRSSTVAE_MODEL_DIR (nullopt
    // when neither: the published artifacts).
    std::optional<std::string> model() const;

private:
    std::string precision_;
    std::optional<std::string> model_;
    checkpoint::Fetcher fetcher_;
    mutable std::mutex mu_;   // held for a load (seconds) and to read what it left
    std::shared_ptr<const OpenDecoder> decoder_;
    std::optional<std::string> error_;
};

// The listener's renderer for cfg: a ListenerRenderer of its precision and
// model, or null when cfg.render is false.
std::shared_ptr<live::Renderer> listener_renderer(const live::LiveConfig& cfg,
                                                  checkpoint::Fetcher fetcher = nullptr);

}  // namespace sstvae::qrss::codec

#endif  // SSTVAE_QRSS_CODEC_LISTENER_RENDER_HPP
