#include "qrss/codec/codec.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "codec/codec.hpp"
#include "images/images.hpp"
#include "qrss/pytext.hpp"

namespace sstvae::qrss::codec {

namespace fs = std::filesystem;

namespace {

// Python's whitespace as int() strips it from an ASCII string: str.isspace()
// (0x09-0x0D, 0x1C-0x1F, 0x20), which int() maps to ' ' and then strips.
bool py_space(char c) {
    const auto u = static_cast<unsigned char>(c);
    return u == 0x20 || (u >= 0x09 && u <= 0x0D) || (u >= 0x1C && u <= 0x1F);
}

int digit_value(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'z') return c - 'a' + 10;
    if (c >= 'A' && c <= 'Z') return c - 'A' + 10;
    return 99;
}

// UTF-8 of one code point (repr() output only ever holds valid ones).
void put_utf8(std::string& out, char32_t cp) {
    if (cp < 0x80) {
        out += static_cast<char>(cp);
    } else if (cp < 0x800) {
        out += static_cast<char>(0xC0 | (cp >> 6));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        out += static_cast<char>(0xE0 | (cp >> 12));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        out += static_cast<char>(0xF0 | (cp >> 18));
        out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    }
}

bool ends_with(std::string_view s, std::string_view tail) {
    return s.size() >= tail.size() && s.compare(s.size() - tail.size(), tail.size(), tail) == 0;
}

// pathlib's split: (root, parts) of an already normalised path string.
std::pair<std::string, std::vector<std::string>> split_path(std::string_view text) {
    std::string root;
    std::string_view rest = text;
    if (!rest.empty() && rest[0] == '/') {
        if (rest.size() >= 2 && rest[1] == '/' && !(rest.size() >= 3 && rest[2] == '/')) {
            root = "//";
            rest.remove_prefix(2);
        } else {
            root = "/";
            rest.remove_prefix(1);
        }
    }
    std::vector<std::string> parts;
    std::size_t i = 0;
    while (i <= rest.size()) {
        const std::size_t j = std::min(rest.find('/', i), rest.size());
        const std::string_view part = rest.substr(i, j - i);
        if (!part.empty() && part != ".") parts.emplace_back(part);
        i = j + 1;
    }
    return {root, parts};
}

std::string join_path(const std::string& root, const std::vector<std::string>& parts) {
    std::string out = root;
    for (std::size_t k = 0; k < parts.size(); ++k) {
        if (k) out += '/';
        out += parts[k];
    }
    return out.empty() ? "." : out;
}

// Path(dir) / name, as str() prints it.
std::string child(const std::string& dir, const std::string& name) {
    auto [root, parts] = split_path(dir);
    parts.push_back(name);
    return join_path(root, parts);
}

// PurePath.name and .suffix.
std::string path_name(const std::string& p) {
    const auto [root, parts] = split_path(p);
    return parts.empty() ? std::string() : parts.back();
}

std::string path_suffix(const std::string& p) {
    const std::string name = path_name(p);
    const std::size_t dot = name.rfind('.');
    // A leading dot is not a suffix; neither is a trailing one (".onnx" and
    // "a." have none).
    if (dot == std::string::npos || dot == 0 || dot + 1 == name.size()) return {};
    return name.substr(dot);
}

// PurePath.with_name(name).
std::string with_name(const std::string& p, const std::string& name) {
    auto [root, parts] = split_path(p);
    if (parts.empty()) throw std::invalid_argument(p + " has an empty name");
    parts.back() = name;
    return join_path(root, parts);
}

bool is_dir(const std::string& p) {
    std::error_code ec;
    return fs::is_directory(fs::path(p), ec);
}

bool exists(const std::string& p) {
    std::error_code ec;
    return fs::exists(fs::path(p), ec);
}

// The names of a directory's entries (pathlib's glob sees them all, dot
// files and directories included), byte-sorted as Python sorts them.
std::vector<std::string> entry_names(const std::string& dir) {
    std::vector<std::string> out;
    std::error_code ec;
    for (fs::directory_iterator it(fs::path(dir), ec), end; !ec && it != end; it.increment(ec))
        out.push_back(it->path().filename().string());
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

// One environment for the metadata sessions, never destroyed: onnxruntime's
// environment is a process-wide singleton, and tearing it down during
// static destruction races whatever else holds it.
Ort::Env& ort_env() {
    static Ort::Env* env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "sstvae-qrss");
    return *env;
}

}  // namespace

// --- Python's text conventions --------------------------------------------------------

std::optional<PyInt> py_int(std::string_view text, int base) {
    if (base != 10 && base != 16) throw std::invalid_argument("py_int: base must be 10 or 16");
    std::size_t i = 0, n = text.size();
    while (i < n && py_space(text[i])) ++i;
    while (n > i && py_space(text[n - 1])) --n;
    bool negative = false;
    if (i < n && (text[i] == '+' || text[i] == '-')) negative = text[i++] == '-';
    if (base == 16 && i + 1 < n && text[i] == '0' && (text[i + 1] == 'x' || text[i + 1] == 'X')) {
        i += 2;
        if (i < n && text[i] == '_') ++i;   // one underscore may follow the prefix
    }
    std::vector<int> digits;
    bool after_underscore = false;
    for (; i < n; ++i) {
        const char c = text[i];
        if (c == '_') {
            if (digits.empty() || after_underscore) return std::nullopt;
            after_underscore = true;
            continue;
        }
        const int d = digit_value(c);
        if (d >= base) return std::nullopt;
        digits.push_back(d);
        after_underscore = false;
    }
    if (digits.empty() || after_underscore) return std::nullopt;

    // The value in decimal, schoolbook (Python ints have no limit).
    std::vector<int> dec{0};   // little-endian decimal digits
    for (const int d : digits) {
        int carry = d;
        for (int& x : dec) {
            const int v = x * base + carry;
            x = v % 10;
            carry = v / 10;
        }
        while (carry) {
            dec.push_back(carry % 10);
            carry /= 10;
        }
    }
    while (dec.size() > 1 && dec.back() == 0) dec.pop_back();
    const bool zero = dec.size() == 1 && dec[0] == 0;
    PyInt out;
    if (negative && !zero) out.decimal = "-";
    for (auto it = dec.rbegin(); it != dec.rend(); ++it) out.decimal += static_cast<char>('0' + *it);
    // int64 when it fits: |v| <= 2^63 - 1, or exactly -2^63.
    unsigned long long mag = 0;
    bool fits = dec.size() <= 19;
    if (fits) {
        for (auto it = dec.rbegin(); it != dec.rend(); ++it) mag = mag * 10 + static_cast<unsigned>(*it);
        fits = negative ? mag <= 9223372036854775808ULL : mag <= 9223372036854775807ULL;
    }
    if (fits) {
        if (!negative) out.value = static_cast<std::int64_t>(mag);
        else out.value = mag == 9223372036854775808ULL ? INT64_MIN : -static_cast<std::int64_t>(mag);
    }
    return out;
}

std::string invalid_int_literal(std::string_view text, int base) {
    // "%.200R": the repr, cut to 200 characters.
    const std::u32string cps = pytext::decode(pytext::repr(text));
    std::string shown;
    for (std::size_t k = 0; k < cps.size() && k < 200; ++k) put_utf8(shown, cps[k]);
    return "invalid literal for int() with base " + std::to_string(base) + ": " + shown;
}

std::string pathlib_str(std::string_view text) {
    const auto [root, parts] = split_path(text);
    return join_path(root, parts);
}

// --- locating the model ------------------------------------------------------------------

std::string resolve_onnx(std::string_view part_in, const std::optional<std::string>& path,
                         std::string_view precision_in, const checkpoint::Fetcher& fetcher) {
    const std::string part(part_in);
    const std::string precision(part == checkpoint::GRAD_PART ? checkpoint::GRAD_PRECISION : precision_in);
    const std::string grad(checkpoint::GRAD_PART);
    if (!path) return checkpoint::resolve_onnx(part, {}, precision, fetcher);

    const std::string p = pathlib_str(*path);
    if (is_dir(p)) {
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
        return child(p, matches.front());
    }

    if (path_suffix(p) == ".onnx") {
        // `-decoder-` is a prefix of `-decoder-grad-`, so a plain substring
        // test would hand back the gradient graph when the decoder was asked.
        const std::string name = path_name(p);
        const bool is_grad = name.find("-" + grad + "-") != std::string::npos;
        if (name.find("-" + part + "-") != std::string::npos && is_grad == (part == grad)) return p;
        const std::string other = is_grad ? grad : (part == "encoder" ? "decoder" : "encoder");
        const std::size_t at = name.find("-" + other + "-");
        if (at == std::string::npos)
            throw ModelError("cannot tell which part " + name + " is -- expected a name like v1-" + part + "-" +
                             precision + ".onnx, or pass the directory instead");
        const std::string sibling_name = name.substr(0, at) + "-" + part + "-" + precision + ".onnx";
        const std::string sibling = with_name(p, sibling_name);
        if (!exists(sibling))
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
        Ort::SessionOptions opts;
        // Only the metadata is read; one thread spares a pool nobody uses.
        opts.SetIntraOpNumThreads(1);
        opts.SetInterOpNumThreads(1);
#ifdef _WIN32
        const std::wstring wpath(onnx_path.begin(), onnx_path.end());
        Ort::Session sess(ort_env(), wpath.c_str(), opts);
#else
        Ort::Session sess(ort_env(), onnx_path.c_str(), opts);
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
    const std::optional<PyInt> v = py_int(head, 16);
    if (!v) return std::nullopt;
    return v->value;   // four hex digits always fit
}

std::optional<std::int64_t> codec_id_from_onnx(const std::string& path_or_dir, std::string_view precision) {
    std::string p = pathlib_str(path_or_dir);
    if (is_dir(p)) {
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
