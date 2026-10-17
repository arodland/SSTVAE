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
//   unless --no-store: Store(--store)             (default $QRSSTVAE_HOME or
//         ~/.local/share/qrsstvae) OSError if its directories cannot be made
//     unless --no-passband, and only for a capture at frontend.FE_FS:
//         PassbandStore(--passband or STORE/passband).write_capture(cap)
//                                                 OSError (the directory, meta.json, an
//                                                 hour's files); meta.json's json/KeyError/
//                                                 ValueError; ValueError for a store at
//                                                 another fs; all uncaught
//   receiver.receive_slot(cap, frames, estimator)
//   no pass: "no CE or L signal found", exit 1, with " (slot kept for a later
//         --retro search)" when a store is open and --no-passband is not given
//         (whatever the capture's rate: the script's condition, which does not
//         ask whether the passband store was written)
//   pass_lines(p) for every pass, then (with a store) associate(p, store):
//         "  stored     provisional (<rule>)" or
//         "  stored     <CALL> <pid:08x> by <rule, or 'existing membership'>"
//                                                 the store's errors uncaught (OSError,
//                                                 KeyError, ValueError, ...)
//   --image: render_pass(passes[0], ...)          (receive.hpp's hook)
// --passband and --no-passband with --no-store are accepted and ignored, as
// there. No lock is taken on either store, as the script takes none, so a
// receive runs beside a listener holding live.dir_lock on the same store
// and nothing orders their writes (the Store's mutex is this process's).
//
// The store's exceptions end the run as decode.cpp's and listen.cpp's do:
// the traceback's last line with the class the script's would name
// (OSError's PEP 3151 subclass, KeyError, ValueError, TypeError,
// OverflowError, IndexError), with one difference shared with those two: a
// store file json cannot parse (the passband store's meta.json, an
// accumulator's or a cluster's index) is the script's
// json.decoder.JSONDecodeError and this one's ValueError (its base class),
// with json's message; a damaged .npz in the store ends both with the
// C++ store's ValueError text where the script names np.load's or
// zipfile's class (decode.cpp's note); and a packed passband hour the
// capture is written into that does not decode is soundfile's
// LibsndfileError there and RuntimeError (its base class) here, from main's
// backstop, in the FLAC decoder's words, naming the same file.
//
// --just-ce and --just-l are argparse's mutually exclusive group
// (argparse.hpp's exclusive()), refused and printed as argparse does.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <limits>
#include <memory>
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
#include "math/errors.hpp"
#include "qrss/associate.hpp"
#include "qrss/qrss_config.hpp"
#include "qrss/frame.hpp"
#include "qrss/npz.hpp"
#include "qrss/passband.hpp"
#include "qrss/pypath.hpp"
#include "qrss/pytext.hpp"
#include "qrss/rawfile.hpp"
#include "qrss/receiver.hpp"
#include "qrss/rules.hpp"
#include "qrss/rx_api.hpp"
#include "qrss/simfile.hpp"
#include "qrss/slot.hpp"
#include "qrss/store.hpp"
#include "qrss/types.hpp"
#include "qrss/wavread.hpp"
#include "receive.hpp"
#include "subcommand.hpp"
#include "util/fileio.hpp"

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

// qrss_receive.py's docstring, its --help description, with the program's
// name in the examples and --image's last sentence as this port has it
// (PNG only: receive_image.cpp's refusal).
constexpr const char* DESCRIPTION =
    "Receive the QRSSTVAE passes, CE and L, in one slot recording.\n"
    "\n"
    "    sstvae-qrss receive rx.wav --slot 2026-10-09T06:00Z\n"
    "    sstvae-qrss receive rx.npz --slot 2026-10-09T06:00Z --frame short --no-store\n"
    "    sstvae-qrss receive rx.wav --slot 2026-10-09T06:00Z --just-l\n"
    "    sstvae-qrss receive rx.wav --slot 2026-10-09T06:00Z --start 2026-10-09T05:59:49Z \\\n"
    "        --image pass.png\n"
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
    "Passes are stored and associated in the multi-pass store\n"
    "(--store DIR, default $QRSSTVAE_HOME or ~/.local/share/qrsstvae)\n"
    "unless --no-store. The slot's whole front-end stream is also kept in the\n"
    "48 h passband store (--passband DIR, default STORE/passband; off with\n"
    "--no-passband or --no-store), where `qrss_decode.py --retro` later\n"
    "template-searches it for passes too weak to detect now. --image renders the single pass (FULL frames with a\n"
    "decoded header only) with the codec (--model DIR, --precision).";

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
    ap.exclusive({"--just-ce", "--just-l"});
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

// --- paths, as pathlib prints them -------------------------------------------------------

// path.suffix.lower() == ".npz". ASCII lowering is str.lower's here: no
// other character lowers to one of ".npz"'s.
bool is_npz(const std::string& path) {
    std::string suffix = pypath::pathlib_suffix(path);
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

// f"{v:08x}" / f"{v:04x}": the sign counts toward the width, as Python's
// zero padding has it (f"{-5:08x}" is "-0000005"; only a hand-edited
// store's key can be negative).
std::string hex(std::int64_t v, int width) {
    std::string digits;
    auto u = v < 0 ? 0 - static_cast<std::uint64_t>(v) : static_cast<std::uint64_t>(v);
    do {
        digits.insert(digits.begin(), "0123456789abcdef"[u % 16]);
        u /= 16;
    } while (u);
    const std::size_t want = static_cast<std::size_t>(width) - (v < 0 ? 1 : 0);
    if (digits.size() < want) digits.insert(0, want - digits.size(), '0');
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

// --- the stores' errors ----------------------------------------------------------------------

// The exception being handled, from the store or the passband store, as the
// script's traceback ends (decode.cpp's and listen.cpp's mapping): its last
// line, exit 1. What is not theirs goes on to main's backstop.
int store_failure(std::ostream& err) {
    try {
        throw;
    } catch (const store::KeyError& e) {   // pyjson's: what() is "KeyError: '<key>'"
        err << e.what() << "\n";
        return 1;
    } catch (const npz::FileError& e) {     // the multi-pass store's files
        return uncaught(err, OsError(e.err, e.path));
    } catch (const rawfile::FileError& e) {   // the passband store's
        return uncaught(err, OsError(util::errno_of(e.code()), e.path()));
    } catch (const math::TypeError& e) {
        return uncaught(err, "TypeError", e.what());
    } catch (const std::overflow_error& e) {
        return uncaught(err, "OverflowError", e.what());
    } catch (const std::out_of_range& e) {
        return uncaught(err, "IndexError", e.what());
    } catch (const std::invalid_argument& e) {
        return uncaught(err, "ValueError", e.what());
    }
}

// --- the run ---------------------------------------------------------------------------------

int run(const std::string& prog, const std::vector<std::string>& argv, std::ostream& out, std::ostream& err) {
    const ArgParser ap = parser(prog);
    int status = 0;
    const std::optional<Args> args = ap.parse(argv, out, err, status);
    if (!args) return status;

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
            if (!util::is_dir(parent))   // os.path.isdir: UTF-8 to the wide API on Windows
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
        // the str sys.argv holds (as transmit's --slot): on POSIX a byte
        // that is not UTF-8 is an escape there, which no ISO time accepts
        if (const auto& s = args->text("slot"); s && !s->empty()) {
            try {
                q = slot::parse_slot(pypath::fs_str(*s));
            } catch (const std::invalid_argument& e) {
                return system_exit(err, "--slot " + pytext::repr(pypath::fs_decode(*s)) + ": " + e.what());
            }
        }
        if (const auto& s = args->text("start"); s && !s->empty()) {
            try {
                start = static_cast<double>(slot::parse_utc(pypath::fs_str(*s)).unix_us()) / 1e6;
            } catch (const std::invalid_argument& e) {
                return system_exit(err, "--start " + pytext::repr(pypath::fs_decode(*s)) + ": " + e.what());
            }
        }
    } catch (const std::overflow_error& e) {
        return uncaught(err, "OverflowError", e.what());
    }

    // load_capture
    const std::string path = pypath::pathlib_str(*args->text("input"));
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

    // The multi-pass store, and the slot's front-end stream in the passband
    // store (a capture at the front end's own rate only).
    std::unique_ptr<store::Store> st;
    const bool no_passband = args->on("no_passband");
    if (!args->on("no_store")) {
        try {
            st = std::make_unique<store::Store>(args->text("store"));
            if (!no_passband && cap.fs == config::FE_FS) {
                // Path(args.passband) if args.passband else store.root / "passband"
                const std::optional<std::string>& pb = args->text("passband");
                passband::PassbandStore(pb && !pb->empty() ? pypath::pathlib_str(*pb)
                                                           : pypath::child(st->root(), "passband"))
                    .write_capture(cap);
            }
        } catch (const std::exception&) {
            return store_failure(err);
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
        out << "no " << names << " signal found"
            << (st && !no_passband ? " (slot kept for a later --retro search)" : "") << "\n";
        return 1;
    }
    for (const types::PassResult& p : passes) {
        for (const std::string& line : pass_lines(p)) out << line << "\n";
        if (!st) continue;
        associate::Result a;
        try {
            a = associate::associate(p, *st);   // rule 2 by the C++ soft_header, as the listener's
        } catch (const std::exception&) {
            return store_failure(err);
        }
        if (!a.key)
            out << "  stored     provisional (" << a.rule << ")\n";
        else
            out << "  stored     " << a.key->callsign << " " << hex(a.key->picture_id, 8) << " by "
                << (a.rule.empty() ? "existing membership" : a.rule) << "\n";
    }
    if (image) return receive::image_renderer().render(passes.front(), *image, out, err);
    return 0;
}

const Subcommand::Register reg("receive", "qrss_receive.py", "receive the CE and L passes in a slot recording",
                               &run);

}  // namespace
}  // namespace sstvae::qrss::cli
