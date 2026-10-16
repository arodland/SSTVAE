// The QRSSTVAE chain with no Python anywhere: the sstvae-qrss binary and
// the C++ transmitter, from a picture to a refined picture.
//
//   sstvae-qrss encode pic.png pic.qrsp          (the v5 encoder)
//   api::pass_audio: a strong SHORT pass in slot q (header and callsign
//     windows: the live receiver gets it) and a weak SHORT pass of the same
//     picture in slot q + 1 (-30 dB in 2500 Hz: too weak for the live
//     receiver, which stores no pass of that slot; the reference's
//     test_decode_cli_retro level), at 1500 Hz, into one 8 kHz recording
//     with this file's fixed-seed noise
//   sstvae-qrss listen --wav rec.wav --slot q --frame short --just-ce
//     --store S --state T --no-pictures   (a replay: the passband store
//     and the multi-pass store are written as live)
//   sstvae-qrss decode --list --store S        the picture is there
//   sstvae-qrss decode out.png --key K --store S --retro --frame short
//     the passband store's search finds the weak pass and attaches it
//   sstvae-qrss decode out2.png --key K --store S --em 1
//     one round of leave-one-out EM over both members
// each step checked for its exit status and the lines that matter, the
// PNGs for being written. Last, the binary is checked for not linking
// libpython (the strings a dynamic link to it would leave in the file).
//
// Built in every tree (the glob), it runs where the binary has the codec:
// elsewhere, or without the models ($QRSSTVAE_MODEL_DIR), it says so and
// passes, unless SSTVAE_REQUIRE_CODEC=1, where that is a failure.
// The binary is <build>/sstvae-qrss beside tests/, or $SSTVAE_QRSS_BIN.
// The SNRs and the weak pass's frame can be moved for an experiment
// ($QRSS_CHAIN_STRONG_DB, $QRSS_CHAIN_WEAK_DB, $QRSS_CHAIN_WEAK_FRAME); the
// defaults are what this test is.

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <variant>
#include <vector>

#ifndef _WIN32
#include <sys/wait.h>
#endif

#include "em_api_fixture.hpp"
#include "images/types.hpp"
#include "qrss/api.hpp"
#include "qrss/qrss_config.hpp"
#include "qrss/decode.hpp"
#include "qrss/png.hpp"
#include "qrss/wav.hpp"
#include "qrss_check.hpp"

using namespace sstvae;
namespace api = sstvae::qrss::api;
namespace cfg = sstvae::qrss::config;
namespace fx = sstvae::em_fixture;
namespace fs = std::filesystem;

namespace {

constexpr double STRONG_DB = -6.0;   // the listener's SHORT smoke level (test_qrss_native_listen.py)
constexpr double WEAK_DB = -30.0;
constexpr const char* WEAK_FRAME = "short";

double env_db(const char* name, double otherwise) {
    const char* v = std::getenv(name);
    return v && *v ? std::strtod(v, nullptr) : otherwise;
}

bool require_codec() {
    const char* v = std::getenv("SSTVAE_REQUIRE_CODEC");
    return v && std::string(v) == "1";
}

std::string quoted(const std::string& s) {
#ifdef _WIN32
    return "\"" + s + "\"";
#else
    std::string out = "'";
    for (const char c : s) out += c == '\'' ? std::string("'\\''") : std::string(1, c);
    return out + "'";
#endif
}

std::string slurp(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

struct Run {
    int rc = -1;
    std::string out, err;
    double seconds = 0.0;
};

// The binary with `args`, stdout and stderr to files in `dir`.
Run run(const fs::path& bin, const std::vector<std::string>& args, const fs::path& dir, const std::string& tag) {
    const fs::path o = dir / (tag + ".out"), e = dir / (tag + ".err");
    std::string cmd = quoted(bin.string());
    for (const std::string& a : args) cmd += " " + quoted(a);
    cmd += " > " + quoted(o.string()) + " 2> " + quoted(e.string());
#ifdef _WIN32
    cmd = "\"" + cmd + "\"";   // cmd.exe strips one pair of outer quotes
#endif
    const auto t = std::chrono::steady_clock::now();
    const int status = std::system(cmd.c_str());
    Run r;
    r.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
#ifdef _WIN32
    r.rc = status;
#else
    r.rc = WIFEXITED(status) ? WEXITSTATUS(status) : 128 + (WIFSIGNALED(status) ? WTERMSIG(status) : 0);
#endif
    r.out = slurp(o);
    r.err = slurp(e);
    std::printf("[%s] rc %d in %.1f s\n", tag.c_str(), r.rc, r.seconds);
    return r;
}

bool has(const std::string& text, const std::string& part) { return text.find(part) != std::string::npos; }

void step_ok(const Run& r, const std::string& what) {
    check::equal(r.rc, 0, what + ": exit status");
    if (r.rc != 0) std::fprintf(stderr, "--- %s stdout:\n%s\n--- stderr:\n%s\n", what.c_str(), r.out.c_str(), r.err.c_str());
}

void contains(const Run& r, const std::string& part, const std::string& what) {
    const bool ok = has(r.out, part);
    check::is_true(ok, what);
    if (!ok) std::fprintf(stderr, "   wanted \"%s\" in:\n%s\n", part.c_str(), r.out.c_str());
}

fs::path self_dir(const char* argv0) {
    std::error_code ec;
#ifdef __linux__
    const fs::path exe = fs::read_symlink("/proc/self/exe", ec);
    if (!ec) return exe.parent_path();
#endif
    return fs::absolute(fs::path(argv0), ec).parent_path();
}

// A 640x480 picture with something in it: a gradient, a disc, bars.
images::Picture test_picture() {
    images::Picture img(640, 480);
    for (int y = 0; y < 480; ++y)
        for (int x = 0; x < 640; ++x) {
            const std::size_t i = (static_cast<std::size_t>(y) * 640 + static_cast<std::size_t>(x)) * 3;
            const double dx = x - 320.0, dy = y - 240.0;
            const bool disc = dx * dx + dy * dy < 120.0 * 120.0;
            const bool bar = (x / 40) % 2 == 0 && y > 400;
            img.rgb[i] = static_cast<std::uint8_t>(disc ? 230 : x * 255 / 639);
            img.rgb[i + 1] = static_cast<std::uint8_t>(disc ? 40 : y * 255 / 479);
            img.rgb[i + 2] = static_cast<std::uint8_t>(bar ? 250 : (disc ? 60 : 128));
        }
    return img;
}

// No dynamic link to libpython: the names such a link leaves in the file.
void no_libpython(const fs::path& bin) {
    const std::string bytes = slurp(bin);
    check::is_true(!bytes.empty(), "the binary reads");
    check::is_true(!has(bytes, "libpython"), "sstvae-qrss names no libpython");
    check::is_true(!has(bytes, "Python.framework/"), "sstvae-qrss names no Python.framework");
    bool dll = false;
    for (std::size_t at = bytes.find("python3"); at != std::string::npos; at = bytes.find("python3", at + 1)) {
        std::size_t k = at + 7;
        while (k < bytes.size() && bytes[k] >= '0' && bytes[k] <= '9') ++k;
        if (k + 4 <= bytes.size() && (bytes.compare(k, 4, ".dll") == 0 || bytes.compare(k, 4, ".DLL") == 0)) dll = true;
    }
    check::is_true(!dll, "sstvae-qrss names no python3*.dll");
}

int chain(const fs::path& bin) {
    const char* models = std::getenv("QRSSTVAE_MODEL_DIR");
    const fs::path dir = fs::temp_directory_path() /
                         ("sstvae_test_em_chain_" +
                          std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(dir);
    struct Cleanup {
        fs::path d;
        ~Cleanup() {
            std::error_code ec;
            if (!std::getenv("QRSS_CHAIN_KEEP")) fs::remove_all(d, ec);
        }
    } cleanup{dir};

    // Has the binary the codec, and are the models there?
    check::current_step = "codec";
    const Run help = run(bin, {"encode", "--help"}, dir, "encode-help");
    if (help.rc != 0 || !models || !*models) {
        const std::string why = help.rc != 0 ? "this sstvae-qrss has no encode (built without the codec)"
                                             : "no $QRSSTVAE_MODEL_DIR";
        if (require_codec()) {
            check::fail("chain", why + ", and SSTVAE_REQUIRE_CODEC=1");
            return 1;
        }
        std::printf("skip: %s\n", why.c_str());
        return 0;
    }
    const std::vector<std::string> model = {"--model", models, "--precision", "fp16"};

    // encode
    check::current_step = "encode";
    qrss::png::write_png(test_picture(), (dir / "pic.png").string());
    std::vector<std::string> enc = {"encode", (dir / "pic.png").string(), (dir / "pic.qrsp").string()};
    enc.insert(enc.end(), model.begin(), model.end());
    const Run e = run(bin, enc, dir, "encode");
    step_ok(e, "encode");
    if (e.rc != 0) return 1;
    const api::Source src = api::read_source((dir / "pic.qrsp").string());
    const auto& sp = std::get<qrss::picture::StoredPicture>(src);
    const std::string pid = qrss::decode::py_hex(sp.picture_id, 8);
    const std::string key = "K1ABC:" + pid;

    // the recording: SHORT strong in slot q, the weak pass in slot q + 1
    check::current_step = "transmit";
    const double strong_db = env_db("QRSS_CHAIN_STRONG_DB", STRONG_DB);
    const double weak_db = env_db("QRSS_CHAIN_WEAK_DB", WEAK_DB);
    const char* wf_env = std::getenv("QRSS_CHAIN_WEAK_FRAME");
    const std::string weak_frame = wf_env && *wf_env ? wf_env : WEAK_FRAME;
    const std::int64_t q = api::parse_slot("2025-10-01T00:00Z");
    api::PassRequest strong;
    strong.q = q;
    strong.callsign = "K1ABC";
    strong.grid = "FN42";
    strong.frame = "short";
    api::PassRequest weak = strong;
    weak.q = q + 1;
    weak.frame = weak_frame;
    const std::vector<float> a = api::pass_audio(src, strong);
    const std::vector<float> b = api::pass_audio(src, weak);
    const double start = api::audio_start_s(q);
    const double end = api::audio_end_s(q + 1, api::Waveform::CE, weak_frame) + 30.0;
    const auto n = static_cast<std::size_t>(std::ceil((end - start) * cfg::FS));
    std::vector<double> rec = fx::gauss(2026, n);
    fx::add_pass(rec, a, 0, 1.0, strong_db);
    fx::add_pass(rec, b, static_cast<std::size_t>(std::llround((api::audio_start_s(q + 1) - start) * cfg::FS)), 1.0,
                 weak_db);
    qrss::wav::write_float32((dir / "rec.wav").string(), rec);
    std::printf("recording: %.0f s, SHORT at %+.1f dB in slot %lld, %s at %+.1f dB in slot %lld\n",
                static_cast<double>(n) / cfg::FS, strong_db, static_cast<long long>(q), weak_frame.c_str(), weak_db,
                static_cast<long long>(q + 1));
    rec.clear();
    rec.shrink_to_fit();

    // listen: a replay into a state directory with a store and a passband store
    check::current_step = "listen";
    const std::string store = (dir / "store").string();
    const Run l = run(bin,
                      {"listen", "--wav", (dir / "rec.wav").string(), "--slot", api::slot_iso(q), "--frame", "short",
                       "--just-ce", "--store", store, "--state", (dir / "state").string(), "--no-pictures"},
                      dir, "listen");
    step_ok(l, "listen");
    check::is_true(fs::is_directory(dir / "store" / "passband"), "listen: the passband store was kept");

    // decode --list: the picture, one member
    check::current_step = "decode --list";
    const Run li = run(bin, {"decode", "--list", "--store", store}, dir, "list");
    step_ok(li, "decode --list");
    contains(li, "K1ABC " + pid, "decode --list: the picture is in the store");
    contains(li, "  1 passes", "decode --list: one member, the strong pass");
    std::size_t stored = 0;
    for (const auto& f : fs::directory_iterator(dir / "store" / "passes")) stored += f.path().extension() == ".npz";
    check::equal(stored, std::size_t{1}, "listen: the live receiver stored the strong pass only");

    // decode --retro --frame <the weak pass's>: found and attached
    check::current_step = "decode --retro";
    std::vector<std::string> rr = {"decode", (dir / "out.png").string(), "--key", key, "--store", store, "--retro",
                                   "--frame", weak_frame};
    rr.insert(rr.end(), model.begin(), model.end());
    const Run r = run(bin, rr, dir, "retro");
    step_ok(r, "decode --retro");
    contains(r, "retro: slot q=" + std::to_string(q + 1) + ": Z ", "decode --retro: the weak slot searched");
    contains(r, "attached to K1ABC " + pid + " by corr", "decode --retro: found and attached by correlation");
    contains(r, "retro: 1 pass found", "decode --retro: one pass found");
    contains(r, "2 passes", "decode --retro: the picture has two members now");
    contains(r, "wrote " + (dir / "out.png").string(), "decode --retro: the picture written");
    check::is_true(fs::exists(dir / "out.png") && fs::file_size(dir / "out.png") > 1000, "decode --retro: out.png");

    // decode --em 1: one round over both members
    check::current_step = "decode --em";
    std::vector<std::string> em = {"decode", (dir / "out2.png").string(), "--key", key, "--store", store, "--em", "1"};
    em.insert(em.end(), model.begin(), model.end());
    const Run m = run(bin, em, dir, "em");
    step_ok(m, "decode --em 1");
    contains(m, "round 1 pass ", "decode --em: a pass re-received");
    contains(m, "EM: mean W ", "decode --em: the summary line");
    contains(m, "; W is the receiver's own estimate", "decode --em: and its clause");
    contains(m, "wrote " + (dir / "out2.png").string(), "decode --em: the picture written");
    check::is_true(fs::exists(dir / "out2.png") && fs::file_size(dir / "out2.png") > 1000, "decode --em: out2.png");
    std::printf("--- retro:\n%s--- em:\n%s", r.out.c_str(), m.out.c_str());
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    check::report_crashes_instead_of_prompting();
    check::Watchdog watchdog(900.0, "qrss_em_chain");
    (void)argc;
    try {
        const char* env = std::getenv("SSTVAE_QRSS_BIN");
#ifdef _WIN32
        const char* name = "sstvae-qrss.exe";
#else
        const char* name = "sstvae-qrss";
#endif
        const fs::path bin = env && *env ? fs::path(env) : self_dir(argv[0]).parent_path() / name;
        check::current_step = "binary";
        if (!fs::exists(bin)) {
            check::fail("chain", "no binary at " + bin.string());
            return check::report("qrss_em_chain");
        }
        no_libpython(bin);
        chain(bin);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL in %s: %s\n", check::current_step.load(), e.what());
        return 1;
    }
    return check::report("qrss_em_chain");
}
