#include "qrss/codec/codec.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "codec/codec.hpp"
#include "images/images.hpp"
#include "qrss/pynum.hpp"
#include "qrss/pypath.hpp"
#include "qrss/pytext.hpp"
#include "util/fileio.hpp"

namespace sstvae::qrss::codec {

namespace {

bool ends_with(std::string_view s, std::string_view tail) {
    return s.size() >= tail.size() && s.compare(s.size() - tail.size(), tail.size(), tail) == 0;
}

// The names of a directory's entries (pathlib's glob sees them all, dot
// files and directories included), byte-sorted as Python sorts them.
std::vector<std::string> entry_names(const std::string& dir) {
    std::error_code ec;
    std::vector<std::string> out = util::list_dir(dir, ec);
    std::sort(out.begin(), out.end());
    return out;
}

std::string join(const std::vector<std::string>& items, const char* sep) {
    std::string out;
    for (std::size_t i = 0; i < items.size(); ++i) {
        if (i) out += sep;
        out += items[i];
    }
    return out;
}

// --- onnxruntime -------------------------------------------------------------

// StatusCodeToString and the Python binding's exception classes, by code.
struct OrtName {
    const char* status;
    const char* python_class;
};
constexpr std::array<OrtName, 16> ORT_NAMES{{
    {"SUCCESS", "Fail"},
    {"FAIL", "Fail"},
    {"INVALID_ARGUMENT", "InvalidArgument"},
    {"NO_SUCHFILE", "NoSuchFile"},
    {"NO_MODEL", "NoModel"},
    {"ENGINE_ERROR", "EngineError"},
    {"RUNTIME_EXCEPTION", "RuntimeException"},
    {"INVALID_PROTOBUF", "InvalidProtobuf"},
    {"MODEL_LOADED", "ModelLoaded"},
    {"NOT_IMPLEMENTED", "NotImplemented"},
    {"INVALID_GRAPH", "InvalidGraph"},
    {"EP_FAIL", "EPFail"},
    {"MODEL_LOAD_CANCELED", "ModelLoadCanceled"},
    {"MODEL_REQUIRES_COMPILATION", "ModelRequiresCompilation"},
    {"NOT_FOUND", "NotFound"},
    {"DEVICE_RESET", "DeviceReset"},
}};

OrtError ort_error(const Ort::Exception& e) {
    const int code = static_cast<int>(e.GetOrtErrorCode());
    const bool known = code >= 0 && static_cast<std::size_t>(code) < ORT_NAMES.size();
    const OrtName n = known ? ORT_NAMES[static_cast<std::size_t>(code)] : OrtName{"GENERAL ERROR", "Fail"};
    return OrtError(code, n.python_class,
                    "[ONNXRuntimeError] : " + std::to_string(code) + " : " + n.status + " : " + e.what());
}

}  // namespace

// --- locating the model ------------------------------------------------------------------

std::string resolve_onnx(std::string_view part_in, const std::optional<std::string>& path,
                         std::string_view precision_in, const checkpoint::Fetcher& fetcher) {
    const std::string part(part_in);
    const std::string precision(part == checkpoint::GRAD_PART ? checkpoint::GRAD_PRECISION : precision_in);
    const std::string grad(checkpoint::GRAD_PART);
    if (!path) return checkpoint::resolve_onnx(part, {}, precision, fetcher);

    const std::string p = pathlib_str(*path);
    if (util::is_dir(p)) {
        const std::string tail = "-" + part + "-" + precision + ".onnx";
        std::vector<std::string> matches, available;
        for (const std::string& name : entry_names(p)) {
            if (ends_with(name, tail)) matches.push_back(name);
            if (ends_with(name, ".onnx")) available.push_back(name);
        }
        if (matches.empty())
            throw ModelError("no *" + tail + " in " + p + "\n" +
                             (available.empty() ? std::string("that directory holds no .onnx files at all")
                                                : "found: " + join(available, ", ")));
        if (matches.size() > 1)
            throw ModelError("ambiguous: " + std::to_string(matches.size()) + " candidates for " + part + "/" +
                             precision + " in " + p + " (" + join(matches, ", ") + ")");
        return pypath::child(p, matches.front());
    }

    if (pypath::pathlib_suffix(p) == ".onnx") {
        // `-decoder-` is a prefix of `-decoder-grad-`, so a plain substring
        // test would hand back the gradient graph when the decoder was asked.
        const std::string name = pypath::pathlib_name(p);
        const bool is_grad = name.find("-" + grad + "-") != std::string::npos;
        if (name.find("-" + part + "-") != std::string::npos && is_grad == (part == grad)) return p;
        const std::string other = is_grad ? grad : (part == "encoder" ? "decoder" : "encoder");
        const std::size_t at = name.find("-" + other + "-");
        if (at == std::string::npos)
            throw ModelError("cannot tell which part " + name + " is -- expected a name like v1-" + part + "-" +
                             precision + ".onnx, or pass the directory instead");
        const std::string sibling_name = name.substr(0, at) + "-" + part + "-" + precision + ".onnx";
        const std::string sibling = pypath::with_name(p, sibling_name);
        if (!util::exists(sibling))
            throw ModelError("need the " + part + " as well as the " + other + ", but " + sibling_name +
                             " is not next to " + name);
        return sibling;
    }

    throw ModelError("--model " + *path +
                     ": expected a .pt checkpoint, a .onnx artifact, or a directory containing the exported "
                     ".onnx files");
}

// --- the codec ID -----------------------------------------------------------------------

std::optional<std::string> source_sha256(const std::string& onnx_path) {
    try {
        // An environment per read, released with the session, as SSTVAE's
        // codec holds one per codec: onnxruntime refcounts one process-wide
        // environment. One left alive at exit (it used to be a leaked
        // static) aborts onnxruntime 1.22's teardown on macOS x86_64 with
        // "mutex lock failed".
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "sstvae-qrss");
        Ort::SessionOptions opts;
        // Only the metadata is read; one thread spares a pool nobody uses.
        opts.SetIntraOpNumThreads(1);
        opts.SetInterOpNumThreads(1);
#ifdef _WIN32
        // onnxruntime takes a wide path here: the UTF-8 name as UTF-16, never
        // widened byte by byte (which turns "\xc3\xa9" into two characters).
        const std::optional<std::wstring> wpath = util::wide_of(onnx_path);
        if (!wpath)
            throw OrtError(3, "NoSuchFile", "[ONNXRuntimeError] : 3 : NO_SUCHFILE : Load model from " + onnx_path +
                                                " failed: the name is not UTF-8");
        Ort::Session sess(env, wpath->c_str(), opts);
#else
        Ort::Session sess(env, onnx_path.c_str(), opts);
#endif
        Ort::AllocatorWithDefaultOptions alloc;
        Ort::ModelMetadata meta = sess.GetModelMetadata();
        auto value = meta.LookupCustomMetadataMapAllocated("sstvae.source_sha256", alloc);
        if (!value) return std::nullopt;
        return std::string(value.get());
    } catch (const Ort::Exception& e) {
        throw ort_error(e);
    }
}

std::optional<std::int64_t> codec_id_of_sha(std::string_view sha) {
    // if not sha or len(sha) < 4: None; int(sha[:4], 16), None on ValueError.
    const std::u32string cps = pytext::decode(sha);
    if (cps.size() < 4) return std::nullopt;
    std::string head;
    for (std::size_t k = 0; k < 4; ++k) {
        if (cps[k] >= 0x80) return std::nullopt;   // see py_int: ASCII only
        head += static_cast<char>(cps[k]);
    }
    const std::optional<pynum::PyInt> v = pynum::py_int(head, 16);
    if (!v) return std::nullopt;
    return v->value;   // four hex digits always fit
}

std::optional<std::int64_t> codec_id_from_onnx(const std::string& path_or_dir, std::string_view precision) {
    std::string p = pathlib_str(path_or_dir);
    if (util::is_dir(p)) {
        std::optional<std::string> found;
        for (const char* part : {"encoder", "decoder"}) {
            try {
                found = resolve_onnx(part, p, precision);
                break;
            } catch (const ModelError&) {
                continue;
            }
        }
        if (!found)
            throw NoModelFile("no *-encoder-" + std::string(precision) + ".onnx or *-decoder-" +
                              std::string(precision) + ".onnx in " + path_or_dir);
        p = *found;
    }
    const std::optional<std::string> sha = source_sha256(p);
    if (!sha) return std::nullopt;
    return codec_id_of_sha(*sha);
}

// --- the encode ----------------------------------------------------------------------------

Encoder onnx_encoder(const std::optional<std::string>& model, std::string_view precision,
                     const checkpoint::Fetcher& fetcher) {
    auto codec = std::make_shared<::sstvae::codec::OnnxCodec>(
        [model, prec = std::string(precision), fetcher](const std::string& part) {
            return resolve_onnx(part, model, prec, fetcher);
        });
    return [codec](const images::ImageArray& img) {
        try {
            return codec->encode(img);
        } catch (const Ort::Exception& e) {
            throw ort_error(e);
        }
    };
}

images::ImageArray load_image(const std::string& path) { return images::load_array(path); }

picture::StoredPicture encode_picture(const std::string& image_path, std::int64_t mode, std::int64_t codec_id,
                                      const Encoder& encode) {
    // Refused before the encode rather than after it (the reference's
    // from_latents would refuse the same, with the same text).
    picture::check_mode(mode);
    picture::check_codec_id(codec_id);
    return encode_picture(load_image(image_path), mode, codec_id, encode);
}

}  // namespace sstvae::qrss::codec
