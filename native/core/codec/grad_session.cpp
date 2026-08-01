#include "codec/grad_session.hpp"

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "config.hpp"
#include "images/types.hpp"
#include "latents/latents.hpp"

namespace sstvae::codec {

struct GradSession::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_ERROR, "sstvae-grad"};
    Ort::MemoryInfo mem =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Session session{nullptr};
    std::vector<float> target;
    std::vector<std::string> in_names;
    std::vector<std::string> out_names;

    explicit Impl(const std::string& path) {
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(4);  // as in codec.cpp; measured best there
        opts.SetLogSeverityLevel(3);
#ifdef _WIN32
        const std::wstring wpath(path.begin(), path.end());
        session = Ort::Session(env, wpath.c_str(), opts);
#else
        session = Ort::Session(env, path.c_str(), opts);
#endif
        Ort::AllocatorWithDefaultOptions alloc;
        // By position, not by name. The artifact names its inputs after
        // the decoder's convention ("latents", "weights", "target") and
        // the prototype export used different words; position is the
        // part that is contractual.
        for (std::size_t i = 0; i < session.GetInputCount(); ++i) {
            in_names.emplace_back(session.GetInputNameAllocated(i, alloc).get());
        }
        for (std::size_t i = 0; i < session.GetOutputCount(); ++i) {
            out_names.emplace_back(session.GetOutputNameAllocated(i, alloc).get());
        }
        if (in_names.size() != 3 || out_names.size() != 3) {
            throw std::runtime_error(
                path + " is not a decoder-gradient graph (expected 3 inputs and 3 "
                       "outputs: latents/weights/target -> recon/grad_z/mse, got " +
                std::to_string(in_names.size()) + " and " +
                std::to_string(out_names.size()) + ")");
        }
    }
};

GradSession::GradSession(const std::string& artifact, const images::ImageArray& target)
    : impl_(std::make_unique<Impl>(artifact)) {
    if (target.width != images::IMG_W || target.height != images::IMG_H) {
        throw std::runtime_error("latent optimization wants a full-size target picture");
    }
    impl_->target = target.chw;
}

GradSession::~GradSession() = default;
GradSession::GradSession(GradSession&&) noexcept = default;
GradSession& GradSession::operator=(GradSession&&) noexcept = default;

optimize::GradFn GradSession::fn() {
    Impl* impl = impl_.get();
    return [impl](const std::vector<float>& latents, const std::vector<float>& weights,
                  std::vector<float>& grad, double& mse) {
        const std::size_t n = static_cast<std::size_t>(latents::N_LATENTS);
        if (latents.size() != n || weights.size() != n) {
            throw std::runtime_error("gradient graph wants full-length (mode C) vectors");
        }
        // ORT wants non-const pointers into the caller's buffers; it
        // does not write through them for an input tensor.
        auto* z = const_cast<float*>(latents.data());
        auto* w = const_cast<float*>(weights.data());

        const std::array<std::int64_t, 4> zshape{1, config::LATENT_CHANNELS,
                                                 config::LATENT_H, config::LATENT_W};
        const std::array<std::int64_t, 4> tshape{1, 3, images::IMG_H, images::IMG_W};
        Ort::Value ins[3] = {
            Ort::Value::CreateTensor<float>(impl->mem, z, n, zshape.data(), zshape.size()),
            Ort::Value::CreateTensor<float>(impl->mem, w, n, zshape.data(), zshape.size()),
            Ort::Value::CreateTensor<float>(impl->mem, impl->target.data(),
                                            impl->target.size(), tshape.data(),
                                            tshape.size()),
        };

        const char* in_names[3] = {impl->in_names[0].c_str(), impl->in_names[1].c_str(),
                                   impl->in_names[2].c_str()};
        const char* out_names[3] = {impl->out_names[0].c_str(),
                                    impl->out_names[1].c_str(),
                                    impl->out_names[2].c_str()};

        auto out = impl->session.Run(Ort::RunOptions{nullptr}, in_names, ins, 3,
                                     out_names, 3);
        // Outputs are (recon, grad_z, mse) in that order -- the export
        // names them and the order is part of the artifact's contract,
        // checked by the input/output count above.
        const float* g = out[1].GetTensorData<float>();
        const std::size_t ng = out[1].GetTensorTypeAndShapeInfo().GetElementCount();
        if (ng != n) {
            throw std::runtime_error("gradient graph returned " + std::to_string(ng) +
                                     " values, expected " + std::to_string(n));
        }
        grad.assign(g, g + ng);
        mse = static_cast<double>(*out[2].GetTensorData<float>());
    };
}

}  // namespace sstvae::codec
