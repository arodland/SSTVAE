#include "codec/codec.hpp"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace sstvae::codec {

namespace {

// numpy's `.round()` and Python's `round()` are half-to-even, and so is
// `nearbyint` under the default rounding mode. `std::round` is not --
// it rounds half away from zero, which is a different function.
inline int quantize(float v) {
    // In *float32*, exactly as numpy does it. NEP 50 keeps
    // `float32_array * python_int` at float32, so computing the product
    // in double instead moves a handful of values across the .5
    // boundary and changes the picture. Measured: 3 differing subpixels
    // in 921600 with a double multiply, 0 with this one.
    const int q = static_cast<int>(std::nearbyintf(v * 255.0f));
    return std::clamp(q, 0, 255);
}

}  // namespace

struct OnnxCodec::Impl {
    Resolver resolver;
    Ort::Env env{ORT_LOGGING_LEVEL_ERROR, "sstvae"};
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::map<std::string, Ort::Session> sessions;
    // part -> the checkpoint sha256 its artifact was exported from.
    std::map<std::string, std::string> sources;

    Ort::Session& session(const std::string& part) {
        auto it = sessions.find(part);
        if (it != sessions.end()) return it->second;
        if (!resolver) {
            throw std::runtime_error("codec has no resolver; cannot locate the " +
                                     part + " artifact");
        }
        const std::string path = resolver(part);

        Ort::SessionOptions opts;
        // Measured best of 1/4/24 on x86-64; 1 was ~5x worse. The
        // receive loop reconstructs once per poll, so this is about not
        // being gratuitously slow rather than about latency.
        opts.SetIntraOpNumThreads(4);
        opts.SetLogSeverityLevel(3);

#ifdef _WIN32
        // ORT takes a wide path on Windows. The artifacts live under a
        // user profile, which can be non-ASCII.
        const std::wstring wpath(path.begin(), path.end());
        Ort::Session sess(env, wpath.c_str(), opts);
#else
        Ort::Session sess(env, path.c_str(), opts);
#endif
        auto [pos, _] = sessions.emplace(part, std::move(sess));
        check_same_checkpoint(part, path, pos->second);
        return pos->second;
    }

    // Both parts must come from one training run.
    //
    // An encoder and decoder from different checkpoints are not a
    // codec: they will load, run, produce a picture, and the picture
    // will be garbage with nothing anywhere reporting a problem. That
    // is the worst failure available here and it is entirely
    // avoidable, because the exporter stamps each artifact with its
    // source checkpoint's sha256.
    //
    // Precisions may differ freely -- an fp16 encoder with an int8
    // decoder is the same codec. Only the checkpoint has to agree.
    void check_same_checkpoint(const std::string& part, const std::string& path,
                               Ort::Session& sess) {
        Ort::AllocatorWithDefaultOptions alloc;
        Ort::ModelMetadata meta = sess.GetModelMetadata();
        auto value = meta.LookupCustomMetadataMapAllocated("sstvae.source_sha256", alloc);
        if (!value) return;  // third-party or hand-rolled export; nothing to check
        const std::string sha(value.get());
        sources[part] = sha;
        for (const auto& [other, other_sha] : sources) {
            if (other_sha == sha) continue;
            throw std::runtime_error(
                part + " and " + other + " come from different checkpoints (" +
                sha.substr(0, 12) + " vs " + other_sha.substr(0, 12) + ").\n" + path +
                "\nis not paired with the " + other + " in use. Both halves must be "
                "exported from the same training run, or the decoded picture will be "
                "silently wrong.");
        }
    }
};

OnnxCodec::OnnxCodec(Resolver resolver) : impl_(std::make_unique<Impl>()) {
    impl_->resolver = std::move(resolver);
}
OnnxCodec::~OnnxCodec() = default;
OnnxCodec::OnnxCodec(OnnxCodec&&) noexcept = default;
OnnxCodec& OnnxCodec::operator=(OnnxCodec&&) noexcept = default;

void OnnxCodec::set_resolver(Resolver resolver) { impl_->resolver = std::move(resolver); }

void OnnxCodec::preload(const std::string& part) { (void)impl_->session(part); }

std::vector<double> OnnxCodec::encode(const ImageArray& image) {
    if (image.width != IMG_W || image.height != IMG_H) {
        throw std::runtime_error("encoder wants " + std::to_string(IMG_W) + "x" +
                                 std::to_string(IMG_H) + ", got " +
                                 std::to_string(image.width) + "x" +
                                 std::to_string(image.height));
    }
    const std::size_t n = static_cast<std::size_t>(3) * IMG_H * IMG_W;
    if (image.chw.size() != n) throw std::runtime_error("image array is the wrong size");

    Ort::Session& sess = impl_->session("encoder");
    Ort::AllocatorWithDefaultOptions alloc;
    auto in_name = sess.GetInputNameAllocated(0, alloc);
    auto out_name = sess.GetOutputNameAllocated(0, alloc);
    const char* ins[] = {in_name.get()};
    const char* outs[] = {out_name.get()};

    std::vector<float> buf = image.chw;
    std::array<std::int64_t, 4> shape{1, 3, IMG_H, IMG_W};
    Ort::Value tensor = Ort::Value::CreateTensor<float>(impl_->mem, buf.data(), buf.size(),
                                                       shape.data(), shape.size());
    auto out = sess.Run(Ort::RunOptions{nullptr}, ins, &tensor, 1, outs, 1);

    const float* z = out[0].GetTensorData<float>();
    const std::size_t nz = out[0].GetTensorTypeAndShapeInfo().GetElementCount();
    if (nz != static_cast<std::size_t>(N_LATENTS)) {
        throw std::runtime_error("encoder returned " + std::to_string(nz) +
                                 " latents, expected " + std::to_string(N_LATENTS));
    }
    // latents_to_flat is a pure reshape: the groups are contiguous
    // slices of the channel axis, each flattened C-order, so
    // concatenating them is the same memory in the same order.
    return std::vector<double>(z, z + nz);
}

Picture OnnxCodec::decode(const std::vector<double>& latents,
                          const std::vector<double>& weights) {
    if (latents.size() != static_cast<std::size_t>(N_LATENTS) ||
        weights.size() != latents.size()) {
        throw std::runtime_error("decode wants full-length (mode C) latent and weight "
                                 "vectors; pad_to_full a shorter mode first");
    }
    Ort::Session& sess = impl_->session("decoder");

    std::vector<float> z(latents.size()), w(weights.size());
    for (std::size_t i = 0; i < latents.size(); ++i) {
        w[i] = static_cast<float>(weights[i]);
        // Erased latents must be zeroed, not merely down-weighted.
        z[i] = w[i] > 0 ? static_cast<float>(latents[i]) : 0.0f;
    }

    std::array<std::int64_t, 4> shape{1, config::LATENT_CHANNELS, config::LATENT_H,
                                      config::LATENT_W};
    Ort::Value ins[2] = {
        Ort::Value::CreateTensor<float>(impl_->mem, z.data(), z.size(), shape.data(),
                                        shape.size()),
        Ort::Value::CreateTensor<float>(impl_->mem, w.data(), w.size(), shape.data(),
                                        shape.size()),
    };

    Ort::AllocatorWithDefaultOptions alloc;
    auto n0 = sess.GetInputNameAllocated(0, alloc);
    auto n1 = sess.GetInputNameAllocated(1, alloc);
    auto no = sess.GetOutputNameAllocated(0, alloc);
    const char* in_names[] = {n0.get(), n1.get()};
    const char* out_names[] = {no.get()};

    auto out = sess.Run(Ort::RunOptions{nullptr}, in_names, ins, 2, out_names, 1);
    const float* pic = out[0].GetTensorData<float>();

    // (1, 3, H, W) planar -> (H, W, 3) interleaved uint8.
    Picture p(IMG_W, IMG_H);
    const std::size_t plane = static_cast<std::size_t>(IMG_H) * IMG_W;
    for (int y = 0; y < IMG_H; ++y) {
        for (int x = 0; x < IMG_W; ++x) {
            const std::size_t src = static_cast<std::size_t>(y) * IMG_W + x;
            const std::size_t dst = src * 3;
            for (int c = 0; c < 3; ++c) {
                p.rgb[dst + c] = static_cast<std::uint8_t>(quantize(pic[c * plane + src]));
            }
        }
    }
    return p;
}

}  // namespace sstvae::codec
