#include "qrss/codec/listener_render.hpp"

#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <string_view>
#include <utility>

#include "util/fileio.hpp"

#if defined(_WIN32)
#include <cwchar>
#endif

namespace sstvae::qrss::codec {

namespace {

// render.model_dir(): $QRSSTVAE_MODEL_DIR, else None (an empty value is None too).
std::optional<std::string> model_dir() {
#if defined(_WIN32)
    const wchar_t* v = ::_wgetenv(L"QRSSTVAE_MODEL_DIR");
    if (!v || !*v) return std::nullopt;
    return util::utf8_of(std::wstring_view(v));
#else
    const char* v = std::getenv("QRSSTVAE_MODEL_DIR");
    if (!v || !*v) return std::nullopt;
    return std::string(v);
#endif
}

}  // namespace

ListenerRenderer::ListenerRenderer(std::string precision, std::optional<std::string> model,
                                   checkpoint::Fetcher fetcher)
    : precision_(std::move(precision)), model_(std::move(model)), fetcher_(std::move(fetcher)) {}

std::optional<std::string> ListenerRenderer::model() const { return model_ ? model_ : model_dir(); }

std::optional<std::string> ListenerRenderer::load(std::optional<std::int64_t> codec_id, const Warn& warn) {
    std::optional<std::string> warning;
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (decoder_) return std::nullopt;
        if (error_) return error_;   // never tried again
        try {
            LoadedDecoder d = codec::load(precision_, model(), codec_id, false, fetcher_);
            decoder_ = std::make_shared<const OpenDecoder>(std::move(d.decoder));
            warning = std::move(d.warning);
        } catch (const std::exception& e) {
            // str(e) or type(e).__name__ (every refusal here has a text)
            const std::string what = e.what();
            error_ = what.empty() ? live::error_name(e) : what;
            return error_;
        }
    }
    if (warning && warn) warn(*warning);   // (outside the lock: warn is the listener's log)
    return std::nullopt;
}

std::optional<std::string> ListenerRenderer::check(std::optional<std::int64_t> codec_id) {
    std::shared_ptr<const OpenDecoder> d;
    {
        std::lock_guard<std::mutex> lk(mu_);
        d = decoder_;
    }
    if (!d) return std::nullopt;
    try {
        // (a decoder without the metadata only warns, which _render does not show)
        check_codec_id(d->codec_id, codec_id);
    } catch (const CodecMismatch& e) {
        return std::string(e.what());
    }
    return std::nullopt;
}

images::Picture ListenerRenderer::render(std::span<const double> S, std::span<const double> W, std::int64_t mode,
                                         std::optional<std::int64_t> /*codec_id*/) {
    std::shared_ptr<const OpenDecoder> d;
    {
        std::lock_guard<std::mutex> lk(mu_);
        d = decoder_;
    }
    if (!d) throw std::runtime_error("the codec is not loaded");
    return codec::render(*d, S, W, mode);
}

bool ListenerRenderer::loaded() const {
    std::lock_guard<std::mutex> lk(mu_);
    return decoder_ != nullptr;
}

std::optional<std::int64_t> ListenerRenderer::decoder_codec_id() const {
    std::lock_guard<std::mutex> lk(mu_);
    return decoder_ ? decoder_->codec_id : std::nullopt;
}

std::optional<std::string> ListenerRenderer::error() const {
    std::lock_guard<std::mutex> lk(mu_);
    return error_;
}

std::shared_ptr<live::Renderer> listener_renderer(const live::LiveConfig& cfg, checkpoint::Fetcher fetcher) {
    if (!cfg.render) return nullptr;
    return std::make_shared<ListenerRenderer>(cfg.precision, cfg.model, std::move(fetcher));
}

}  // namespace sstvae::qrss::codec
