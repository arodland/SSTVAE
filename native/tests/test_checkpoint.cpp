// Finding the model artifacts.
//
// No network anywhere here, and none needed: everything below the
// download itself is path arithmetic against a temporary cache
// (`SSTVAE_MODEL_CACHE`) and a stub `Fetcher`. The download is checked
// by running the CLI against the real Hub, which is not something to
// put in ctest.
//
// The error messages get as much attention as the happy paths, because
// `docs/native-app.md` makes the first-run and offline story this
// phase's deliverable: on a field laptop with no connectivity, the
// message *is* the recovery path.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "check.hpp"
#include "checkpoint/checkpoint.hpp"

using namespace sstvae;
namespace fs = std::filesystem;

namespace {

// A scratch directory that also serves as the model cache, so nothing
// here can touch a real one.
class TempDir {
public:
    TempDir() {
        // Portable and unique without getpid(), which needs <unistd.h>
        // on POSIX and does not exist on Windows at all.
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        dir_ = fs::temp_directory_path() /
               ("sstvae-ckpt-" + std::to_string(stamp) + "-" +
                std::to_string(counter_++));
        fs::create_directories(dir_);
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(dir_, ec);
    }
    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;

    const fs::path& path() const { return dir_; }
    std::string str() const { return dir_.string(); }

    fs::path touch(const std::string& name) const {
        const fs::path p = dir_ / name;
        std::ofstream(p) << "not really an onnx file";
        return p;
    }

private:
    fs::path dir_;
    static inline int counter_ = 0;
};

void set_cache(const std::string& dir) {
#if defined(_WIN32)
    _putenv_s("SSTVAE_MODEL_CACHE", dir.c_str());
#else
    setenv("SSTVAE_MODEL_CACHE", dir.c_str(), 1);
#endif
}

// Runs `fn` and returns the CheckpointError text, or "" if it did not
// throw one.
template <typename F>
std::string error_text(F&& fn) {
    try {
        fn();
    } catch (const checkpoint::CheckpointError& e) {
        return e.what();
    } catch (...) {
        return "<wrong exception type>";
    }
    return "";
}

bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

// --- names ------------------------------------------------------------------

void test_filenames() {
    check::equal(checkpoint::onnx_filename("encoder", "fp16"),
                 std::string("v1-encoder-fp16.onnx"), "ckpt/name: encoder fp16");
    check::equal(checkpoint::onnx_filename("decoder", "int8"),
                 std::string("v1-decoder-int8.onnx"), "ckpt/name: decoder int8");
    check::equal(checkpoint::onnx_filename("decoder"),
                 std::string("v1-decoder-fp16.onnx"),
                 "ckpt/name: fp16 is the default precision");

    // The revision is the checkpoint stem, so the artifacts can never be
    // bumped independently of the checkpoint they came from.
    check::is_true(contains(checkpoint::onnx_filename("encoder"),
                            std::string(checkpoint::DEFAULT_REVISION)),
                   "ckpt/name: carries the revision");

    check::is_true(contains(error_text([] { checkpoint::onnx_filename("codec"); }),
                            "encoder"),
                   "ckpt/name: a bad part is refused and the options named");
    const std::string bad_precision =
        error_text([] { checkpoint::onnx_filename("encoder", "fp8"); });
    check::is_true(contains(bad_precision, "fp32") && contains(bad_precision, "int8"),
                   "ckpt/name: a bad precision lists the real ones");
}

void test_url_names_the_repo_and_file() {
    const std::string url = checkpoint::artifact_url("v1-decoder-fp16.onnx");
    check::is_true(contains(url, std::string(checkpoint::DEFAULT_REPO)),
                   "ckpt/url: names the repo");
    check::is_true(contains(url, "v1-decoder-fp16.onnx"), "ckpt/url: names the file");
    check::is_true(url.rfind("https://", 0) == 0, "ckpt/url: https");
}

// --- the cache --------------------------------------------------------------

void test_cache_dir_is_overridable() {
    TempDir tmp;
    set_cache(tmp.str());
    check::equal(checkpoint::cache_dir(), tmp.str(),
                 "ckpt/cache: SSTVAE_MODEL_CACHE wins, so tests never touch a real one");
}

void test_find_cached() {
    TempDir tmp;
    set_cache(tmp.str());
    check::is_true(!checkpoint::find_cached("v1-decoder-fp16.onnx").has_value(),
                   "ckpt/cache: nothing in an empty cache");
    tmp.touch("v1-decoder-fp16.onnx");
    check::is_true(checkpoint::find_cached("v1-decoder-fp16.onnx").has_value(),
                   "ckpt/cache: found once present");
}

void test_a_cache_hit_never_reaches_the_fetcher() {
    // The immutability argument in one assertion: a published filename
    // names specific bytes, so a hit needs no revalidation and the
    // network is not touched at all.
    TempDir tmp;
    set_cache(tmp.str());
    tmp.touch("v1-decoder-fp16.onnx");

    int calls = 0;
    const checkpoint::Fetcher counting = [&](std::string_view) -> std::string {
        ++calls;
        return "";
    };
    const std::string got = checkpoint::resolve_onnx("decoder", "", "fp16", counting);
    check::equal(calls, 0, "ckpt/cache: a hit does not call the fetcher");
    check::equal(fs::path(got).filename().string(), std::string("v1-decoder-fp16.onnx"),
                 "ckpt/cache: and returns the cached file");
}

void test_a_miss_fetches_once() {
    TempDir tmp;
    set_cache(tmp.str());
    std::vector<std::string> asked;
    const checkpoint::Fetcher fake = [&](std::string_view name) -> std::string {
        asked.emplace_back(name);
        return (tmp.path() / std::string(name)).string();
    };
    checkpoint::resolve_onnx("encoder", "", "fp16", fake);
    check::equal(asked.size(), std::size_t{1}, "ckpt/fetch: asked for exactly one file");
    check::equal(asked.front(), std::string("v1-encoder-fp16.onnx"),
                 "ckpt/fetch: the part that was actually needed");
}

void test_no_fetcher_explains_the_way_out() {
    // A build with no networking, or a GUI that has not installed one.
    TempDir tmp;
    set_cache(tmp.str());
    const std::string msg =
        error_text([] { checkpoint::resolve_onnx("decoder", "", "fp16", nullptr); });
    check::is_true(contains(msg, "v1-decoder-fp16.onnx"),
                   "ckpt/offline: names the missing artifact");
    check::is_true(contains(msg, "https://huggingface.co/"),
                   "ckpt/offline: gives the URL to fetch by hand");
    check::is_true(contains(msg, tmp.str()),
                   "ckpt/offline: and where to put it so the next run finds it");
}

void test_a_failing_fetcher_still_explains_the_way_out() {
    // The regression that prompted this check: the fetcher's own
    // CheckpointError used to be rethrown unchanged, which dropped the
    // offline advice entirely -- the one thing an operator needs.
    TempDir tmp;
    set_cache(tmp.str());
    const checkpoint::Fetcher broken = [](std::string_view) -> std::string {
        throw checkpoint::CheckpointError("could not fetch: network unreachable");
    };
    const std::string msg = error_text(
        [&] { checkpoint::resolve_onnx("decoder", "", "fp16", broken); });
    check::is_true(contains(msg, "network unreachable"),
                   "ckpt/offline: keeps the underlying reason");
    check::is_true(contains(msg, "https://huggingface.co/"),
                   "ckpt/offline: *and* still says what to do about it");
}

// --- --model resolution -----------------------------------------------------

void test_a_directory_is_searched() {
    TempDir tmp;
    tmp.touch("v1-encoder-fp16.onnx");
    tmp.touch("v1-decoder-fp16.onnx");
    check::equal(fs::path(checkpoint::resolve_onnx("decoder", tmp.str())).filename().string(),
                 std::string("v1-decoder-fp16.onnx"), "ckpt/dir: picks the right part");
    check::equal(fs::path(checkpoint::resolve_onnx("encoder", tmp.str())).filename().string(),
                 std::string("v1-encoder-fp16.onnx"), "ckpt/dir: and the other one");
}

void test_a_directory_without_the_part_says_what_is_there() {
    TempDir tmp;
    tmp.touch("v1-encoder-fp16.onnx");
    const std::string msg =
        error_text([&] { checkpoint::resolve_onnx("decoder", tmp.str()); });
    check::is_true(contains(msg, "v1-encoder-fp16.onnx"),
                   "ckpt/dir: lists what the directory does hold");

    TempDir empty;
    const std::string none =
        error_text([&] { checkpoint::resolve_onnx("decoder", empty.str()); });
    check::is_true(contains(none, "no .onnx files at all"),
                   "ckpt/dir: an empty directory says so plainly");
}

void test_a_directory_with_two_candidates_is_ambiguous() {
    // Two exports of the same part, e.g. from different checkpoints.
    // Picking one silently would decode with a codec the operator did
    // not choose.
    TempDir tmp;
    tmp.touch("v1-decoder-fp16.onnx");
    tmp.touch("v2-decoder-fp16.onnx");
    const std::string msg =
        error_text([&] { checkpoint::resolve_onnx("decoder", tmp.str()); });
    check::is_true(contains(msg, "ambiguous"), "ckpt/dir: refuses to guess");
    check::is_true(contains(msg, "v1-decoder-fp16.onnx") &&
                       contains(msg, "v2-decoder-fp16.onnx"),
                   "ckpt/dir: and names both candidates");
}

void test_precision_selects_within_a_directory() {
    TempDir tmp;
    tmp.touch("v1-decoder-fp16.onnx");
    tmp.touch("v1-decoder-int8.onnx");
    check::equal(
        fs::path(checkpoint::resolve_onnx("decoder", tmp.str(), "int8")).filename().string(),
        std::string("v1-decoder-int8.onnx"), "ckpt/dir: precision picks the file");
}

void test_a_single_file_is_used_directly() {
    TempDir tmp;
    const fs::path dec = tmp.touch("v1-decoder-fp16.onnx");
    check::equal(checkpoint::resolve_onnx("decoder", dec.string()), dec.string(),
                 "ckpt/file: the named part is used as given");
}

void test_the_sibling_part_is_derived() {
    // `--model v1-encoder-fp16.onnx` must still work for an operation
    // that turns out to need the decoder too.
    TempDir tmp;
    const fs::path enc = tmp.touch("v1-encoder-fp16.onnx");
    tmp.touch("v1-decoder-fp16.onnx");
    check::equal(fs::path(checkpoint::resolve_onnx("decoder", enc.string())).filename().string(),
                 std::string("v1-decoder-fp16.onnx"),
                 "ckpt/file: the sibling is found beside it");
}

void test_a_missing_sibling_is_reported() {
    TempDir tmp;
    const fs::path enc = tmp.touch("v1-encoder-fp16.onnx");
    const std::string msg =
        error_text([&] { checkpoint::resolve_onnx("decoder", enc.string()); });
    check::is_true(contains(msg, "v1-decoder-fp16.onnx"),
                   "ckpt/file: names the sibling it wanted");
    check::is_true(contains(msg, "not next to"), "ckpt/file: and where it looked");
}

void test_an_unrecognisable_name_is_reported() {
    TempDir tmp;
    const fs::path odd = tmp.touch("model.onnx");
    const std::string msg =
        error_text([&] { checkpoint::resolve_onnx("decoder", odd.string()); });
    check::is_true(contains(msg, "cannot tell which part"),
                   "ckpt/file: an .onnx with no part in its name is refused");
    check::is_true(contains(msg, "directory instead"),
                   "ckpt/file: and suggests what to do");
}

void test_something_that_is_neither_is_reported() {
    TempDir tmp;
    const fs::path pt = tmp.touch("v1.pt");
    const std::string msg =
        error_text([&] { checkpoint::resolve_onnx("decoder", pt.string()); });
    check::is_true(contains(msg, "expected a .onnx"),
                   "ckpt/file: a .pt is not something this port can use");

    const std::string missing = error_text(
        [&] { checkpoint::resolve_onnx("decoder", "/no/such/path/at/all.onnx"); });
    check::is_true(!missing.empty(), "ckpt/file: a nonexistent path is an error");
}

}  // namespace

int main() {
    try {
        test_filenames();
        test_url_names_the_repo_and_file();
        test_cache_dir_is_overridable();
        test_find_cached();
        test_a_cache_hit_never_reaches_the_fetcher();
        test_a_miss_fetches_once();
        test_no_fetcher_explains_the_way_out();
        test_a_failing_fetcher_still_explains_the_way_out();
        test_a_directory_is_searched();
        test_a_directory_without_the_part_says_what_is_there();
        test_a_directory_with_two_candidates_is_ambiguous();
        test_precision_selects_within_a_directory();
        test_a_single_file_is_used_directly();
        test_the_sibling_part_is_derived();
        test_a_missing_sibling_is_reported();
        test_an_unrecognisable_name_is_reported();
        test_something_that_is_neither_is_reported();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return check::report("checkpoint");
}
