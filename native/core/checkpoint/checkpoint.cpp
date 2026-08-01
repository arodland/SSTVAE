#include "checkpoint/checkpoint.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace sstvae::checkpoint {

namespace fs = std::filesystem;

namespace {

std::mutex g_fetcher_mutex;
Fetcher g_fetcher;  // NOLINT(cert-err58-cpp) - plain default construction

std::optional<std::string> env(const char* name) {
    const char* v = std::getenv(name);
    if (v == nullptr || *v == '\0') return std::nullopt;
    return std::string(v);
}

fs::path home() {
#if defined(_WIN32)
    if (const auto p = env("USERPROFILE")) return fs::path(*p);
    return fs::path(".");
#else
    if (const auto p = env("HOME")) return fs::path(*p);
    return fs::path(".");
#endif
}

bool is_valid_part(std::string_view part) {
    return part == "encoder" || part == "decoder" || part == GRAD_PART;
}

std::string parts_text() { return "'encoder', 'decoder' or 'decoder-grad'"; }

// The gradient graph exists at one precision only; see GRAD_PRECISION.
std::string_view effective_precision(std::string_view part,
                                     std::string_view precision) {
    return part == GRAD_PART ? GRAD_PRECISION : precision;
}

bool revision_has_grad() {
    return std::find(GRAD_REVISIONS.begin(), GRAD_REVISIONS.end(),
                     DEFAULT_REVISION) != GRAD_REVISIONS.end();
}

bool is_valid_precision(std::string_view precision) {
    return std::find(PRECISIONS.begin(), PRECISIONS.end(), precision) !=
           PRECISIONS.end();
}

std::string precisions_text() {
    std::string out;
    for (std::size_t i = 0; i < PRECISIONS.size(); ++i) {
        if (i != 0) out += ", ";
        out += std::string(PRECISIONS[i]);
    }
    return out;
}

std::string join(const std::vector<std::string>& items, const char* sep) {
    std::string out;
    for (std::size_t i = 0; i < items.size(); ++i) {
        if (i != 0) out += sep;
        out += items[i];
    }
    return out;
}

// What to tell someone whose machine cannot reach the Hub. Named
// because this text is a deliverable rather than an afterthought: on a
// field laptop it is the whole of the recovery path.
std::string offline_advice(std::string_view filename) {
    return "\nIf you are offline, fetch it on another machine from\n  " +
           artifact_url(filename) + "\nthen either pass --model with that file, or drop it in\n  " +
           cache_dir() + "\nwhere it will be found on the next run.";
}

}  // namespace

std::string onnx_filename(std::string_view part, std::string_view precision) {
    if (!is_valid_part(part)) {
        throw CheckpointError("part must be " + parts_text() + ", not '" +
                              std::string(part) + "'");
    }
    if (!is_valid_precision(precision)) {
        throw CheckpointError("precision must be one of " + precisions_text() +
                              ", not '" + std::string(precision) + "'");
    }
    return std::string(DEFAULT_REVISION) + "-" + std::string(part) + "-" +
           std::string(effective_precision(part, precision)) + ".onnx";
}

std::string cache_dir() {
    if (const auto override_dir = env("SSTVAE_MODEL_CACHE")) return *override_dir;
#if defined(_WIN32)
    if (const auto local = env("LOCALAPPDATA")) {
        return (fs::path(*local) / "sstvae" / "cache" / "models").string();
    }
    return (home() / "AppData" / "Local" / "sstvae" / "cache" / "models").string();
#elif defined(__APPLE__)
    return (home() / "Library" / "Caches" / "sstvae" / "models").string();
#else
    if (const auto xdg = env("XDG_CACHE_HOME")) {
        return (fs::path(*xdg) / "sstvae" / "models").string();
    }
    return (home() / ".cache" / "sstvae" / "models").string();
#endif
}

std::optional<std::string> find_cached(std::string_view filename) {
    std::error_code ec;
    const fs::path p = fs::path(cache_dir()) / std::string(filename);
    if (fs::is_regular_file(p, ec)) return p.string();
    return std::nullopt;
}

std::string artifact_url(std::string_view filename) {
    return "https://huggingface.co/" + std::string(DEFAULT_REPO) + "/resolve/main/" +
           std::string(filename);
}

Fetcher default_fetcher() {
    std::lock_guard<std::mutex> lock(g_fetcher_mutex);
    return g_fetcher;
}

void set_default_fetcher(Fetcher fetcher) {
    std::lock_guard<std::mutex> lock(g_fetcher_mutex);
    g_fetcher = std::move(fetcher);
}

namespace {

// The published artifact: cache first, network only if it is missing.
std::string fetch_published(std::string_view part, std::string_view precision,
                            const Fetcher& fetcher) {
    const std::string filename = onnx_filename(part, precision);
    if (const auto cached = find_cached(filename)) return *cached;

    const Fetcher& fn = fetcher ? fetcher : default_fetcher();
    if (!fn) {
        throw CheckpointError("no " + filename + " in " + cache_dir() +
                              ", and this build cannot download it." +
                              offline_advice(filename));
    }
    try {
        return fn(filename);
    } catch (const std::exception& e) {
        // Every failure gets the advice appended, including a
        // CheckpointError from the fetcher itself. Rethrowing those
        // unchanged -- to avoid double-wrapping -- silently dropped the
        // one piece of text that tells an offline operator what to do,
        // which is this phase's actual deliverable.
        throw CheckpointError(std::string(e.what()) + offline_advice(filename));
    }
}

}  // namespace

std::string resolve_onnx(std::string_view part, std::string_view path,
                         std::string_view precision, const Fetcher& fetcher) {
    if (!is_valid_part(part)) {
        throw CheckpointError("part must be " + parts_text() + ", not '" +
                              std::string(part) + "'");
    }
    precision = effective_precision(part, precision);
    if (path.empty()) {
        if (part == GRAD_PART && !revision_has_grad()) {
            throw CheckpointError(
                "latent optimization needs the " + std::string(GRAD_PART) +
                " artifact, which was not published for " +
                std::string(DEFAULT_REVISION) + " (it exists from " +
                std::string(GRAD_REVISIONS.front()) +
                " onward).\nPass --model with a revision that has it, or a "
                "directory of exported .onnx files.");
        }
        return fetch_published(part, precision, fetcher);
    }

    std::error_code ec;
    const fs::path p{std::string(path)};

    if (fs::is_directory(p, ec)) {
        const std::string suffix =
            "-" + std::string(part) + "-" + std::string(precision) + ".onnx";
        std::vector<std::string> matches;
        std::vector<std::string> any_onnx;
        for (const fs::directory_entry& entry : fs::directory_iterator(p, ec)) {
            const std::string name = entry.path().filename().string();
            if (name.size() < 5 || name.compare(name.size() - 5, 5, ".onnx") != 0) {
                continue;
            }
            any_onnx.push_back(name);
            if (name.size() > suffix.size() &&
                name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
                matches.push_back(entry.path().string());
            }
        }
        std::sort(matches.begin(), matches.end());
        std::sort(any_onnx.begin(), any_onnx.end());

        if (matches.empty()) {
            throw CheckpointError(
                "no *-" + std::string(part) + "-" + std::string(precision) +
                ".onnx in " + p.string() + "\n" +
                (any_onnx.empty() ? "that directory holds no .onnx files at all"
                                  : "found: " + join(any_onnx, ", ")));
        }
        if (matches.size() > 1) {
            std::vector<std::string> names;
            names.reserve(matches.size());
            for (const std::string& m : matches) {
                names.push_back(fs::path(m).filename().string());
            }
            throw CheckpointError("ambiguous: " + std::to_string(matches.size()) +
                                  " candidates for " + std::string(part) + "/" +
                                  std::string(precision) + " in " + p.string() + " (" +
                                  join(names, ", ") + ")");
        }
        return matches.front();
    }

    if (p.extension() == ".onnx") {
        const std::string name = p.filename().string();
        const std::string mine = "-" + std::string(part) + "-";
        const std::string grad_tag = "-" + std::string(GRAD_PART) + "-";
        // `-decoder-` is a prefix of `-decoder-grad-`, so a bare
        // substring test hands back the gradient graph when the decoder
        // was asked for -- and it loads, and its first output *is* a
        // reconstruction, so the mistake survives to a wrong picture
        // rather than an error.
        const bool is_grad = name.find(grad_tag) != std::string::npos;
        if (name.find(mine) != std::string::npos && is_grad == (part == GRAD_PART)) {
            return p.string();
        }

        // Named a different part: derive this one, so `--model
        // v1-encoder-fp16.onnx` still works for an operation that turns
        // out to need the decoder too.
        const std::string other_part =
            is_grad ? std::string(GRAD_PART)
                    : (part == "encoder" ? "decoder" : "encoder");
        const std::string theirs = "-" + other_part + "-";
        const std::size_t at = name.find(theirs);
        if (at == std::string::npos) {
            throw CheckpointError("cannot tell which part " + name +
                                  " is -- expected a name like " +
                                  std::string(DEFAULT_REVISION) + "-" +
                                  std::string(part) + "-" + std::string(precision) +
                                  ".onnx, or pass the directory instead");
        }
        // Rebuild from the stem rather than substituting into the given
        // name: the sibling's *precision* need not match the one we
        // were handed. `decoder-grad` is published at fp32 only, so the
        // gradient sibling of an fp16 encoder is an fp32 name that a
        // substitution would never produce.
        const std::string sibling_name = name.substr(0, at) + mine +
                                         std::string(precision) + ".onnx";
        const fs::path sibling = p.parent_path() / sibling_name;
        if (!fs::is_regular_file(sibling, ec)) {
            throw CheckpointError("need the " + std::string(part) + " as well as the " +
                                  other_part + ", but " + sibling_name +
                                  " is not next to " + name);
        }
        return sibling.string();
    }

    throw CheckpointError("--model " + std::string(path) +
                          ": expected a .onnx artifact or a directory containing "
                          "the exported .onnx files");
}

}  // namespace sstvae::checkpoint
