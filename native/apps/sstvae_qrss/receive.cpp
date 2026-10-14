// sstvae-qrss receive: qrss_receive.py. Every QRSSTVAE pass, CE and L, in
// one slot recording (an 8 kHz WAV, or a qrss_simulate.py .npz), received
// and reported.
//
// The script, step by step, with how it ends at each one:
//   rules.apply(--old-edge-noise, --old-gating)   (the environment's
//         QRSS_OLD_EDGE_NOISE / QRSS_OLD_GATING count as well, as there)
//   parse_slot(--slot) if --slot                  ValueError -> SystemExit("--slot '<text>': <why>")
//   parse_time(--start) if --start                ValueError -> SystemExit("--start '<text>': <why>")
//   load_capture: not a file                      SystemExit("<path>: no such file")
//         .npz: channel.load_npz                  uncaught (np.load's, zipfile's, KeyError, ...)
//         WAV without --slot                      SystemExit("--slot is required for a WAV input")
//         WAV: receiver.capture_from_wav          uncaught (OSError, ValueError)
//   select_frames: an .npz frame the flags exclude, or a frame with no twin: SystemExit
//   receiver.receive_slot(cap, frames, estimator)
//   no pass: "no CE or L signal found", exit 1
//   pass_lines(p) for every pass
//   --image: render_pass(passes[0], ...)          (receive.hpp's hook)
//
// Stage 2 of the port has no multi-pass store (plan stage 3), so only
// --no-store is served: without it, and with --store, --passband or
// --no-passband (which only mean anything to the store), this refuses
// before reading anything rather than receive without storing.
//
// --just-ce and --just-l are argparse's mutually exclusive group, which
// argparse.hpp does not model: the conflict is found here, at the argument
// argparse would raise it at, and the usage line shows the group as
// argparse prints it.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <limits>
#include <optional>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "argparse.hpp"
#include "common.hpp"
#include "qrss/config.hpp"
#include "qrss/frame.hpp"
#include "qrss/pytext.hpp"
#include "qrss/receiver.hpp"
#include "qrss/rules.hpp"
#include "qrss/rx_api.hpp"
#include "qrss/simfile.hpp"
#include "qrss/slot.hpp"
#include "qrss/types.hpp"
#include "qrss/wavread.hpp"
#include "receive.hpp"
#include "subcommand.hpp"

namespace sstvae::qrss::cli {

namespace receive {

namespace {
ImageRenderer& renderer_slot() {
    static ImageRenderer r;   // set during static initialisation, read-only after
    return r;
}
}  // namespace

ImageRenderer image_renderer() { return renderer_slot(); }

RegisterImageRenderer::RegisterImageRenderer(ImageRenderer r) { renderer_slot() = r; }

}  // namespace receive

namespace {

namespace fr = sstvae::qrss::frame;

// The Python class read_wav's refusal is raised as (wavread.hpp), by the
// name a traceback's last line gives it.
const char* wav_error_class(wav::FormatError::Kind k) {
    using K = wav::FormatError::Kind;
    switch (k) {
        case K::struct_error: return "struct.error";
        case K::zero_division: return "ZeroDivisionError";
        case K::type: return "TypeError";
        case K::unbound_local: return "UnboundLocalError";
        case K::value: break;
    }
    return "ValueError";
}

// qrss_receive.py's docstring, its --help description, with the store's
// paragraph as this port has it.
constexpr const char* DESCRIPTION =
    "Receive the QRSSTVAE passes, CE and L, in one slot recording.\n"
    "\n"
    "    sstvae-qrss receive rx.wav --slot 2026-10-09T06:00Z --no-store\n"
    "    sstvae-qrss receive rx.npz --slot 2026-10-09T06:00Z --frame short --no-store\n"
    "    sstvae-qrss receive rx.wav --slot 2026-10-09T06:00Z --just-l --no-store\n"
    "    sstvae-qrss receive rx.wav --slot 2026-10-09T06:00Z --start 2026-10-09T05:59:49Z \\\n"
    "        --no-store --image pass.png\n"
    "\n"
    "A WAV is 8 kHz audio starting 12 s before t0 (as `qrss_transmit.py`\n"
    "writes it) unless --start gives its first sample's UTC time; an .npz is\n"
    "`qrss_simulate.py`'s front-end output (its own slot and frame are used\n"
    "unless --slot/--frame are given). Every signal in the slot is found\n"
    "(preamble search, then the whole-slot track-before-detect), verified,\n"
    "tracked and demodulated (`sstvae.qrss.receiver.receive_slot`), and for\n"
    "each pass this prints its waveform and frame, its frequency at t0 (an L\n"
    "pass's is its pilot's), drift, wander, Doppler, clock error, Z_ref, SNR\n"
    "in 2500 Hz, the header, the callsign windows' text and match, the mean\n"
    "weight W in dB and the association result.\n"
    "\n"
    "Both waveforms are listened for by default; --just-ce or --just-l\n"
    "listens for one. A CE-only receive is the receiver's original\n"
    "single-waveform path and knows nothing of L: an L signal in the\n"
    "recording can come out as a weak, headerless CE pass. Two receiver\n"
    "rules changed on 2026-10-12, the noise floor near 300 and 2700 Hz and\n"
    "which candidates are verified around a strong signal\n"
    "(`sstvae.qrss.rules`); --old-edge-noise and --old-gating set them back,\n"
    "and with both --just-ce is the receiver of 2026-10-11 bit for bit. --frame names\n"
    "the frame shape in either waveform (\"short\" or \"l-short\"), and the\n"
    "other waveform listens for the same shape (`lframe.twin`); a frame with\n"
    "no twin (the CE version-1 and test frames) needs the flag that names its\n"
    "own waveform.\n"
    "An .npz's own frame fixes the waveform it holds, so a flag excluding\n"
    "that waveform is refused rather than reporting an empty slot.\n"
    "\n"
    "qrss_receive.py stores and associates passes in the multi-pass store\n"
    "unless --no-store, and keeps the slot's front-end stream in the 48 h\n"
    "passband store. The store is not ported to sstvae-qrss yet (plan stage\n"
    "3): give --no-store; --store, --passband and --no-passband are refused.\n"
    "--image renders the single pass (FULL frames with a decoded header only)\n"
    "with the codec (--model DIR, --precision).";

constexpr const char* STORE_REFUSAL =
    "the multi-pass store is not ported to sstvae-qrss yet (plan stage 3): run with --no-store "
    "(and without --store, --passband or --no-passband), or use qrss_receive.py";

const char* waveform_name(std::int64_t wf) { return wf == config::WAVEFORM_L ? "L" : "CE"; }

std::vector<std::string> frame_choices() {
    // sorted(frame.PRESETS) + sorted(lframe.PRESETS)
    std::vector<std::string> out;
    for (const auto group : {config::PresetGroup::Presets, config::PresetGroup::LPresets}) {
        std::vector<std::string> names;
        for (const fr::FrameSpec& s : fr::presets(group)) names.push_back(s.name());
        std::sort(names.begin(), names.end());
        out.insert(out.end(), names.begin(), names.end());
    }
    return out;
}

ArgParser parser(const std::string& prog) {
    ArgParser ap(prog, DESCRIPTION);
    ap.positional("input", "slot recording: 8 kHz WAV, or a qrss_simulate .npz");
    ap.option("--slot", ArgSpec().described("quarter hour, ISO UTC, e.g. 2026-10-09T06:00Z"));
    ap.option("--start", ArgSpec().described("UTC time of the WAV's first sample (default: t0 - 12 s)"));
    ap.option("--frame", ArgSpec().one_of(frame_choices()).described(
                             "frame shape, either waveform's name for it (default full, or the .npz's own)"));
    ap.flag("--just-ce", "listen for waveform CE only (default: CE and L)");
    ap.flag("--just-l", "listen for waveform L only (default: CE and L)");
    ap.option("--store", ArgSpec().described("multi-pass store directory"));
    ap.flag("--no-store", "do not store or associate");
    ap.option("--passband", ArgSpec().described("48 h passband store directory (default STORE/passband)"));
    ap.flag("--no-passband", "do not keep the slot's front-end stream");
    ap.option("--estimator", ArgSpec().one_of({"joint", "plain"}).otherwise("joint").described(
                                 "per-block data estimator (default joint)"));
    ap.option("--image", ArgSpec().described("render the (first) pass's picture to this PNG"));
    ap.option("--model", ArgSpec().described("codec model directory (for --image)"));
    ap.option("--precision", ArgSpec().one_of({"fp32", "fp16", "int8"}).otherwise("fp32").described(
                                 "codec precision (for --image)"));
    // rules.add_arguments(ap)
    ap.flag("--old-edge-noise", "take the noise floor near 300 and 2700 Hz as before 2026-10-12, partly off "
                                "the front end's filter skirt");
    ap.flag("--old-gating", "verify every candidate as before 2026-10-12, including those inside a signal "
                            "found already");
    return ap;
}

// --- the mutually exclusive group ----------------------------------------------------

constexpr std::string_view GROUP_USAGE_SPLIT = "[--just-ce] [--just-l]";
constexpr std::string_view GROUP_USAGE = "[--just-ce | --just-l]";

// argparse prints a mutually exclusive group's options as one bracket.
std::string group_usage(std::string text) {
    for (std::size_t at = text.find(GROUP_USAGE_SPLIT); at != std::string::npos;
         at = text.find(GROUP_USAGE_SPLIT, at + GROUP_USAGE.size()))
        text.replace(at, GROUP_USAGE_SPLIT.size(), GROUP_USAGE);
    return text;
}

struct Conflict {
    std::size_t at;       // the argument argparse raises the conflict at
    std::string first;    // the group's option seen first ("--just-ce")
    std::string second;   // the other one, at `at`
};

// The parser's long options, read off its usage line (and --help, which
// the usage line shows as -h): what argparse matches an abbreviation
// against.
std::vector<std::string> long_flags(const ArgParser& ap) {
    std::vector<std::string> out = {"--help"};
    std::istringstream words(ap.usage());
    for (std::string w; words >> w;) {
        if (!w.empty() && w.front() == '[') w.erase(0, 1);
        while (!w.empty() && w.back() == ']') w.pop_back();
        if (w.rfind("--", 0) == 0 && w.size() > 2) out.push_back(w);
    }
    return out;
}

// Where argparse's take_action raises "not allowed with": the first
// argument naming the other option of the group once one has been seen.
// Arguments are classified as argparse does: after the first "--" all are
// values; "--opt=x" is never a group flag (a store_true given a value is
// refused before take_action); an exact flag or a unique prefix of one.
std::optional<Conflict> exclusive_conflict(const std::vector<std::string>& argv,
                                           const std::vector<std::string>& flags) {
    std::optional<std::string> seen;
    for (std::size_t i = 0; i < argv.size(); ++i) {
        const std::string& a = argv[i];
        if (a == "--") break;
        if (a.rfind("--", 0) != 0 || a.find('=') != std::string::npos) continue;
        std::string hit;
        if (std::find(flags.begin(), flags.end(), a) != flags.end()) {
            hit = a;
        } else {
            std::vector<std::string> m;
            for (const std::string& f : flags)
                if (f.rfind(a, 0) == 0) m.push_back(f);
            if (m.size() != 1) continue;
            hit = m.front();
        }
        if (hit != "--just-ce" && hit != "--just-l") continue;
        if (!seen) seen = hit;
        else if (*seen != hit) return Conflict{i, *seen, hit};
    }
    return std::nullopt;
}

// parse_args with the group: nullopt after help or a usage error (written
// to out / err, `status` 0 or 2).
std::optional<Args> parse_args(const ArgParser& ap, const std::string& prog, const std::vector<std::string>& argv,
                               std::ostream& out, std::ostream& err, int& status) {
    const std::optional<Conflict> c = exclusive_conflict(argv, long_flags(ap));
    std::ostringstream o, e;
    // Up to the conflict, argparse consumes as it would without one: help
    // and every error raised while consuming come first. Errors raised after
    // consuming (missing and unrecognized arguments) never come, the conflict
    // being raised before.
    const std::vector<std::string> head(argv.begin(), c ? argv.begin() + static_cast<std::ptrdiff_t>(c->at)
                                                        : argv.end());
    std::optional<Args> args = ap.parse(head, o, e, status);
    if (c) {
        const std::string text = e.str();
        const bool late = text.find(": error: the following arguments are required: ") != std::string::npos ||
                          text.find(": error: unrecognized arguments: ") != std::string::npos;
        if (args || late) {
            err << group_usage(ap.usage()) << "\n"
                << prog << ": error: argument " << c->second << ": not allowed with argument " << c->first << "\n";
            status = 2;
            return std::nullopt;
        }
    }
    out << group_usage(o.str());
    err << group_usage(e.str());
    return args;
}

// --- paths, as pathlib prints them -------------------------------------------------------

// str(PurePosixPath(text)): "." for nothing, '.' parts and repeated or
// trailing slashes dropped, exactly two leading slashes kept.
std::string pathlib_str(std::string_view text) {
    std::string root;
    if (!text.empty() && text[0] == '/') {
        root = text.size() >= 2 && text[1] == '/' && !(text.size() >= 3 && text[2] == '/') ? "//" : "/";
        text.remove_prefix(root.size() == 2 ? 2 : 1);
    }
    std::string out = root;
    bool first = true;
    std::size_t i = 0;
    while (i <= text.size()) {
        const std::size_t j = std::min(text.find('/', i), text.size());
        const std::string_view part = text.substr(i, j - i);
        if (!part.empty() && part != ".") {
            if (!first) out += '/';
            out += part;
            first = false;
        }
        i = j + 1;
    }
    return out.empty() ? "." : out;
}

// path.suffix.lower() == ".npz": the last part's final ".npz", in any case,
// with something before the dot.
bool is_npz(const std::string& normalised) {
    const std::size_t slash = normalised.rfind('/');
    const std::string name = slash == std::string::npos ? normalised : normalised.substr(slash + 1);
    const std::size_t dot = name.rfind('.');
    if (dot == std::string::npos || dot == 0 || dot + 1 == name.size()) return false;
    std::string suffix = name.substr(dot);
    std::transform(suffix.begin(), suffix.end(), suffix.begin(),
                   [](char ch) { return ch >= 'A' && ch <= 'Z' ? static_cast<char>(ch - 'A' + 'a') : ch; });
    return suffix == ".npz";
}

// --- pass_lines ------------------------------------------------------------------------------

// Python's format(x, spec) for a float: printf's digits with '.' for the
// point, and NaN unsigned (Python prints "nan" / "+nan" whatever its sign bit).
std::string fmt(const char* spec, double x) { return fr::py_format(x, spec); }

// format(x, "<width><spec>"): right-aligned in `width` columns.
std::string fmt(const char* spec, double x, std::size_t width) {
    std::string s = fmt(spec, x);
    if (s.size() < width) s.insert(0, width - s.size(), ' ');
    return s;
}

// f"{v:08x}" / f"{v:04x}" of a non-negative int.
std::string hex(std::int64_t v, int width) {
    std::string digits;
    auto u = static_cast<std::uint64_t>(v < 0 ? -v : v);
    do {
        digits.insert(digits.begin(), "0123456789abcdef"[u % 16]);
        u /= 16;
    } while (u);
    if (static_cast<int>(digits.size()) < width) digits.insert(0, static_cast<std::size_t>(width) - digits.size(), '0');
    return (v < 0 ? "-" : "") + digits;
}

std::vector<std::string> pass_lines(const types::PassResult& p) {
    const types::TrackReport& r = p.report;
    const std::int64_t wf = p.waveform;
    std::vector<std::string> out;
    const std::string wf_name =
        wf == config::WAVEFORM_CE || wf == config::WAVEFORM_L ? waveform_name(wf) : std::to_string(wf);
    out.push_back("pass " + p.uid + "   " + wf_name + " " + p.frame);
    out.push_back(std::string("  ") + (wf == config::WAVEFORM_L ? "pilot    " : "frequency") + "  " +
                  fmt("%.3f", r.offset_hz, 9) + " Hz at t0   drift " + fmt("%+.3f", r.drift_hz_per_min) +
                  " Hz/min   wander " + fmt("%.3f", r.wander_hz_rms) + " Hz rms");
    out.push_back("  doppler    " + fmt("%.3f", r.doppler_hz) + " Hz   clock " + fmt("%+.1f", r.ppm) +
                  " ppm   Z_ref " + fmt("%.1f", r.z_ref) + "   SNR2500 " + fmt("%+.1f", r.snr2500_db) +
                  " dB   kappa " + fmt("%.2f", r.kappa) + (r.suspect ? "   SUSPECT" : "") +
                  (r.slip_free ? "" : "   phase slips"));
    if (p.header) {
        const header::HeaderFields& h = *p.header;
        out.push_back("  header     " + h.callsign + " " + (h.grid ? *h.grid : std::string("None")) +
                      " picture " + hex(h.picture_id, 8) + " mode " + std::string(1, h.mode >= 0 && h.mode < 3 ? "ABC"[h.mode] : '?') +
                      " segment " + std::to_string(h.segment) + " codec " + hex(h.codec_id, 4));
    } else {
        out.push_back("  header     not decoded");
    }
    if (p.cw) {
        const types::CwIdResult& cw = *p.cw;
        const std::string agrees = !cw.agrees ? "" : (*cw.agrees ? "   agrees" : "   DISAGREES");
        out.push_back("  callsign   " + (cw.text && !cw.text->empty() ? *cw.text : std::string("(not read)")) +
                      "   match Z " + fmt("%.1f", cw.z_match) + "   keying " + cw.keying + agrees);
    }
    out.push_back("  mean W     " + fmt("%+.2f", receiver::mean_w_db(p)) + " dB");
    return out;
}

// --- the run ---------------------------------------------------------------------------------

int run(const std::string& prog, const std::vector<std::string>& argv, std::ostream& out, std::ostream& err) {
    const ArgParser ap = parser(prog);
    int status = 0;
    const std::optional<Args> args = parse_args(ap, prog, argv, out, err, status);
    if (!args) return status;

    // What this port cannot do yet, refused before anything is read.
    if (!args->on("no_store") || args->text("store") || args->text("passband") || args->on("no_passband"))
        return system_exit(err, STORE_REFUSAL);
    std::optional<receive::ImageRequest> image;
    if (const auto& im = args->text("image")) {
        // check_image's first refusal: the directory the picture would be
        // saved in, str(Path(image).parent) as pathlib prints it, missing
        // (the extension is receive_image.cpp's refusal, PNG only). On
        // Windows a backslash separates too, as it does for WindowsPath
        // (which would print it where this prints '/'): split on '/' alone,
        // C:\no\pic.png would have '.' for its directory and pass.
        {
            const std::string& t = *im;
#ifdef _WIN32
            const auto is_sep = [](char c) { return c == '/' || c == '\\'; };
#else
            const auto is_sep = [](char c) { return c == '/'; };
#endif
            // posixpath.splitroot: "//" kept as the root only when exactly two
            const std::size_t rlen = t.empty() || !is_sep(t[0]) ? 0
                                     : (t.size() > 1 && is_sep(t[1]) && !(t.size() > 2 && is_sep(t[2]))) ? 2
                                                                                                          : 1;
            const std::string root = t.substr(0, rlen);
            std::vector<std::string> parts;   // without empty and '.' components
            std::string cur;
            for (std::size_t k = rlen; k <= t.size(); ++k) {
                if (k == t.size() || is_sep(t[k])) {
                    if (!cur.empty() && cur != ".") parts.push_back(cur);
                    cur.clear();
                } else {
                    cur += t[k];
                }
            }
            const auto joined = [&](std::size_t n) {
                std::string out = root;
                for (std::size_t k = 0; k < n; ++k) out += (k ? "/" : "") + parts[k];
                return out.empty() ? std::string(".") : out;
            };
            const std::string path = joined(parts.size());
            const std::string parent = parts.empty() ? path : joined(parts.size() - 1);
            std::error_code ec;
            if (!std::filesystem::is_directory(std::filesystem::path(parent), ec))
                return system_exit(err, "--image " + path + ": " + parent + ": no such directory");
        }
        // render.model_dir(): $QRSSTVAE_MODEL_DIR when --model is absent.
        std::optional<std::string> model = args->text("model");
        if (!model) {
            const char* env = std::getenv("QRSSTVAE_MODEL_DIR");
            if (env && *env) model = env;
        }
        image = receive::ImageRequest{*im, model, *args->text("precision")};
        const receive::ImageRenderer r = receive::image_renderer();
        if (!r.render)
            return system_exit(err, "--image: this sstvae-qrss renders no pictures (the codec half of receive is "
                                    "not compiled in: it needs SSTVAE_BUILD_CODEC and "
                                    "apps/sstvae_qrss/codec/receive_image.cpp); use qrss_receive.py --image");
        if (r.refusal)
            if (const std::optional<std::string> why = r.refusal(*image)) return system_exit(err, *why);
    }

    // rules.apply(args): the flag, or the environment's switch as rules.old reads it.
    api::ReceiveConfig rc;
    rc.old_edge_noise = args->on("old_edge_noise") || rules::old(rules::EDGE_NOISE);
    rc.old_gating = args->on("old_gating") || rules::old(rules::GATING);
    rc.estimator = *args->text("estimator");

    std::optional<std::int64_t> q;
    std::optional<double> start;
    try {
        if (const auto& s = args->text("slot"); s && !s->empty()) {
            try {
                q = slot::parse_slot(*s);
            } catch (const std::invalid_argument& e) {
                return system_exit(err, "--slot " + pytext::repr(*s) + ": " + e.what());
            }
        }
        if (const auto& s = args->text("start"); s && !s->empty()) {
            try {
                start = static_cast<double>(slot::parse_utc(*s).unix_us()) / 1e6;
            } catch (const std::invalid_argument& e) {
                return system_exit(err, "--start " + pytext::repr(*s) + ": " + e.what());
            }
        }
    } catch (const std::overflow_error& e) {
        return uncaught(err, "OverflowError", e.what());
    }

    // load_capture
    const std::string path = pathlib_str(*args->text("input"));
    if (!is_file(path)) return system_exit(err, path + ": no such file");
    api::Capture cap;
    std::optional<std::string> npz_frame;
    if (is_npz(path)) {
        simfile::Sim sim;
        try {
            sim = simfile::parse(read_file(path));
        } catch (const PyError& e) {
            return uncaught(err, e);
        } catch (const simfile::Error& e) {
            return uncaught(err, e.cls(), e.what());
        }
        // Sim(spec=lframe.get_any(str(d["frame"]))): an unknown name is load_npz's ValueError.
        try {
            npz_frame = fr::get_any(sim.frame).name();
        } catch (const std::invalid_argument& e) {
            return uncaught(err, "ValueError", e.what());
        }
        if (sim.fs <= 0 || sim.fs > std::numeric_limits<int>::max())
            return uncaught(err, "ValueError", "simfile: fs " + std::to_string(sim.fs) + " is not a sample rate");
        cap.fe = std::move(sim.fe);
        cap.q = q.value_or(sim.q);
        cap.t0_index = sim.t0_index;
        cap.fs = static_cast<int>(sim.fs);
    } else {
        if (!q) return system_exit(err, "--slot is required for a WAV input");
        try {
            cap = api::capture_from_wav(path, *q, start);
        } catch (const wav::ReadError& e) {
            return uncaught(err, OsError(e.err, e.path));
        } catch (const wav::FormatError& e) {
            return uncaught(err, wav_error_class(e.kind), e.what());
        } catch (const std::invalid_argument& e) {
            return uncaught(err, "ValueError", e.what());
        }
    }

    // select_frames
    std::vector<int> wf;
    if (args->on("just_ce")) wf = {config::WAVEFORM_CE};
    else if (args->on("just_l")) wf = {config::WAVEFORM_L};
    else wf = {config::WAVEFORM_CE, config::WAVEFORM_L};
    const std::optional<std::string>& frame_flag = args->text("frame");
    const fr::FrameSpec& base = fr::get_any(frame_flag ? *frame_flag : npz_frame.value_or("full"));
    if (npz_frame) {
        const int own = static_cast<int>(fr::get_any(*npz_frame).waveform());
        if (std::find(wf.begin(), wf.end(), own) == wf.end())
            return system_exit(err, "the .npz holds frame " + *npz_frame + ", waveform " + waveform_name(own) +
                                        ", which " + (wf.front() == config::WAVEFORM_CE ? "--just-ce" : "--just-l") +
                                        " does not listen for");
    }
    rc.frames.clear();
    for (const int w : wf) {
        try {
            rc.frames.push_back(fr::twin(base, w));
        } catch (const std::invalid_argument&) {
            const bool ce = static_cast<int>(base.waveform()) == config::WAVEFORM_CE;
            return system_exit(err, "frame " + base.name() + " has no " + waveform_name(w) + " twin; add " +
                                        (ce ? "--just-ce" : "--just-l"));
        }
    }

    std::vector<types::PassResult> passes;
    try {
        passes = api::receive_slot(cap, rc);
    } catch (const std::invalid_argument& e) {
        return uncaught(err, "ValueError", e.what());
    }
    // A slot before 1970 (q < 0, from --slot or the .npz): the reference's
    // demod.extract / ldemod.extract_l descramble with sequences.scrambler(q),
    // which refuses q outside uint64, so its receive_slot raises at the first
    // detection that reaches demodulation. receiver::receive_pass refuses it
    // the same way (the catch above); this is only a backstop, since every
    // pass returned has been demodulated.
    if (cap.q < 0 && !passes.empty())
        return uncaught(err, "ValueError", "q must fit in uint64, not " + std::to_string(cap.q));
    if (passes.empty()) {
        std::string names;
        for (const fr::FrameSpec& s : receiver::frames(std::span<const fr::FrameSpec>(rc.frames)))
            names += (names.empty() ? "" : " or ") + std::string(waveform_name(static_cast<int>(s.waveform())));
        out << "no " << names << " signal found\n";
        return 1;
    }
    for (const types::PassResult& p : passes)
        for (const std::string& line : pass_lines(p)) out << line << "\n";
    if (image) return receive::image_renderer().render(passes.front(), *image, out, err);
    return 0;
}

const Subcommand::Register reg("receive", "qrss_receive.py",
                               "receive the CE and L passes in a slot recording (--no-store only)", &run);

}  // namespace
}  // namespace sstvae::qrss::cli
