// sstvae-qrss listen: qrss_listen.py. Listens for QRSSTVAE live, CE and L,
// on 8 kHz audio piped into stdin (or a replayed recording), keeping a tile
// directory current (state.json, tiles/, listener.log) for the desktop
// app's QRSS window. Built on the in-process API, qrss/listen_api.hpp,
// which an application calls directly.
//
// The script, step by step, with how it ends at each one:
//   parse_args                              argparse's refusals (exit 2)
//   build_config: --frame, its L twin,      SystemExit (exit 1), the script's texts
//     --refresh
//   the tile directory's lock               SystemExit "another listener is writing
//                                           tiles to DIR; ..." (exit 1)
//   the store's, the passband store's       printed ("... runs without a store",
//     locks                                 "... keeps none"); it runs on
//   the stores, the listener                OSError for a directory it cannot make
//   --wav: --start, else --slot             SystemExit "--start '<s>': <why>",
//                                           "--slot '<s>': <why>", "--wav needs
//                                           --start or --slot", "<wav>: no such
//                                           file"; the WAV's own errors uncaught
//   stdin a terminal                        SystemExit "pipe 8 kHz mono audio into
//                                           stdin (see --help), or use --wav"
//   "tiles in DIR", then live.run           its log lines on stderr, one per line,
//                                           flushed; exit 0 at the end of the audio
//   Ctrl-C                                  "stopped", state.json not listening, exit 0
//
// How it ends, here:
//   - the end of stdin (or of the replay): "audio ended", state.json not
//     listening, the locks released and exit 0, within about 1.5 s of the
//     end even with a receive in flight, which is abandoned (the script
//     exits under its daemon threads the same way);
//   - SIGINT, and SIGTERM too (the script dies on SIGTERM with nothing
//     written; this stops as on Ctrl-C), or on Windows Ctrl-C, Ctrl-Break
//     and the console closing: "stopped", state.json not listening, exit 0,
//     within the same bound;
//   - an error that escapes the listener (state.json unwritable, say): its
//     line ("OSError: ...", the traceback's last) and exit 1.
// When a stop had to abandon a receive, the process ends with std::_Exit
// once its output is flushed, so the abandoned threads never meet static
// destructors.
//
// What this port does differently, and why:
//   - Without the codec compiled in (codec/listen_render.cpp, with
//     SSTVAE_BUILD_CODEC) the tiles have no pictures: it says so once on
//     stderr and runs as with --no-pictures, rather than refusing a listener
//     whose tiles are useful without them.
//   - The end-of-slot receive runs on a thread, not in a second process.
//   - A read error on stdin ends the audio as its end does, then prints the
//     error's line and exits 1 (the script's reader thread dies with a
//     traceback and the listener waits for audio forever).
//   - A broken stderr (the app gone) loses the lines written to it; SIGPIPE
//     is ignored (the script's print raises there).
//   - A WAV's warnings are printed as "WavFileWarning: <text>" (the
//     script's warnings module prefixes the file and line of their source).

#include <atomic>
#include <cerrno>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#ifdef _WIN32
// Here only (CLAUDE.md: never from a widely-included header).
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

#include "argparse.hpp"
#include "common.hpp"
#include "listen.hpp"
#include "qrss/dirlock.hpp"
#include "qrss/listen_api.hpp"
#include "qrss/listener.hpp"
#include "qrss/npz.hpp"
#include "qrss/passband.hpp"
#include "qrss/pypath.hpp"
#include "qrss/pytext.hpp"
#include "qrss/rawfile.hpp"
#include "qrss/slot.hpp"
#include "qrss/sources.hpp"
#include "qrss/wavread.hpp"
#include "subcommand.hpp"
#include "util/fileio.hpp"

namespace sstvae::qrss::cli {

namespace listen_render {

namespace {
qrss::listen::RendererFactory& factory_slot() {
    static qrss::listen::RendererFactory f;   // set during static initialisation, read-only after
    return f;
}
}  // namespace

qrss::listen::RendererFactory factory() { return factory_slot(); }

Register::Register(qrss::listen::RendererFactory f) { factory_slot() = std::move(f); }

}  // namespace listen_render

namespace {

namespace ql = sstvae::qrss::listen;

// qrss_listen.py's docstring, its --help description, with the program's
// name in the examples, the end-of-slot receive's thread and the FLAC
// packer as this port has them, and a paragraph of its own at the end.
constexpr const char* DESCRIPTION =
    "Listen for QRSSTVAE live: every signal in the passband, of either\n"
    "waveform, becomes a tile whose picture fills in as its pass arrives.\n"
    "\n"
    "    arecord -q -f S16_LE -r 8000 -c 1 -t raw | sstvae-qrss listen --format s16\n"
    "    parec --rate 8000 --channels 1 --format float32le | sstvae-qrss listen\n"
    "    sstvae-qrss listen --wav rx.wav --start 2026-10-09T05:59:49Z    # replay\n"
    "\n"
    "Audio is raw mono 8 kHz on stdin (--format f32, the default, or s16,\n"
    "little-endian), stamped with the computer's clock as it arrives, so the\n"
    "clock must be right to a second or so (NTP). The desktop app's QRSS\n"
    "window starts this with its own capture audio piped in. --wav replays a\n"
    "recording instead (from --start, default 12 s before the slot given by\n"
    "--slot, as `qrss_transmit.py` writes it), as fast as the receiver goes\n"
    "unless --speed is given.\n"
    "\n"
    "Both waveforms are listened for, CE and L (the linear one, for SSB\n"
    "rigs), unless --just-ce or --just-l names one: --frame names the shape,\n"
    "in either waveform's name for it (full or l-full), and each waveform\n"
    "listens for its own frame of that shape, so a CE-only shape such as\n"
    "full-v1 needs --just-ce. --just-ce is the listener as it was before L,\n"
    "which knows nothing of L: an L signal in the slot can come out as a\n"
    "weak, headerless CE tile. The desktop app starts this with no waveform\n"
    "flag, so it hears both. Two receiver rules changed on 2026-10-12, the\n"
    "noise floor near 300 and 2700 Hz and which candidates are verified\n"
    "around a strong signal (`sstvae.qrss.rules`); --old-edge-noise and\n"
    "--old-gating set them back, and with both --just-ce is the listener of\n"
    "2026-10-11 bit for bit.\n"
    "\n"
    "Tiles go to --state DIR (default STORE/live): `state.json`, rewritten\n"
    "after every change, and `tiles/<id>.png`; every log line goes to\n"
    "`listener.log` there too. With rig control the desktop\n"
    "app writes the radio's dial there too (`dial_changes.jsonl`), and a slot\n"
    "heard on more than one dial is received separately on each. A signal is looked for on its\n"
    "preamble (every --refresh seconds until 5 minutes into the slot, then\n"
    "every 5 minutes while none has been found), then received every\n"
    "--refresh seconds (default 2 minutes) with what has arrived so far; when\n"
    "its frame ends the whole slot is received again (round B, and the\n"
    "whole-slot search for signals too weak to see on the preamble) on a\n"
    "second thread, so the other slots' tiles keep updating, stored and\n"
    "associated in the multi-pass store (--store DIR, default $QRSSTVAE_HOME\n"
    "or ~/.local/share/qrsstvae; off with --no-store), and the tile shows the\n"
    "picture accumulated over every pass of it. The front-end stream also\n"
    "goes to the 48 h passband store (--passband DIR, default STORE/passband;\n"
    "off with --no-passband or --no-store) for `qrss_decode.py --retro`.\n"
    "Each finished hour is packed into FLAC (sstvae-qrss's own encoder:\n"
    "nothing to install).\n"
    "\n"
    "A live receive of a FULL slot costs roughly 10-30 s of CPU per signal on\n"
    "a desktop; slow machines want a longer --refresh.\n"
    "\n"
    "sstvae-qrss: Ctrl-C and SIGTERM both stop it as Ctrl-C stops the script\n"
    "(\"stopped\" in the log, state.json not listening). Without the codec\n"
    "compiled in (SSTVAE_BUILD_CODEC), the tiles have no pictures, as with\n"
    "--no-pictures, and it says so once.";

ArgParser parser(const std::string& prog) {
    ArgParser ap(prog, DESCRIPTION);
    ap.option("--format", ArgSpec().one_of({"f32", "s16"}).otherwise("f32").described(
                              "stdin sample format (default f32)"));
    ap.option("--wav", ArgSpec().described("replay this 8 kHz recording"));
    ap.option("--start", ArgSpec().described("UTC time of the replay's first sample"));
    ap.option("--slot", ArgSpec().described("with --wav and no --start: the replay starts 12 s before this slot"));
    ap.option("--speed", ArgSpec().real().otherwise("0.0").described(
                             "replay at this many times real time (default: as fast as possible)"));
    ap.option("--frame", ArgSpec().otherwise("full").described(
                             "frame shape, either waveform's name for it: full or l-full (default full)"));
    ap.flag("--just-ce", "receive waveform CE only (default: both)");
    ap.flag("--just-l", "receive waveform L only (default: both)");
    ap.exclusive({"--just-ce", "--just-l"});
    ap.option("--refresh", ArgSpec().real().otherwise("120.0").described(
                               "seconds between live receives of a slot (default 120)"));
    ap.option("--state", ArgSpec().described("tile directory (default STORE/live)"));
    ap.option("--store", ArgSpec().described("multi-pass store directory"));
    ap.flag("--no-store", "do not store or associate");
    ap.option("--passband", ArgSpec().described("48 h passband store directory"));
    ap.flag("--no-passband", "do not keep the front-end stream");
    ap.option("--recover-hours",
              ArgSpec().real().otherwise("3.0").described(
                  "on start, receive from the passband store every slot that ended this recently while no "
                  "listener finished it (default 3; 0 turns it off)"));
    ap.option("--model", ArgSpec().described("codec model directory"));
    ap.option("--precision", ArgSpec().one_of({"fp32", "fp16", "int8"}).otherwise("fp32").described(
                                 "codec precision (default fp32)"));
    ap.flag("--no-pictures", "tiles without pictures");
    // rules.add_arguments(ap)
    ap.flag("--old-edge-noise", "take the noise floor near 300 and 2700 Hz as before 2026-10-12, partly off "
                                "the front end's filter skirt");
    ap.flag("--old-gating", "verify every candidate as before 2026-10-12, including those inside a signal "
                            "found already");
    return ap;
}

// wavio.read_wav's refusals, as the class the script's traceback names.
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

// Path(path).is_file() as Python 3.13 decides it: false for the errors
// pathlib reads as "not there" (ENOENT, ENOTDIR, EBADF, ELOOP; on Windows
// ERROR_NOT_READY, ERROR_INVALID_NAME, ERROR_CANT_RESOLVE_FILENAME) and for
// a name the OS cannot spell, the OSError for any other.
bool path_is_file(const std::string& path) {
    const auto p = util::fs_path(path);
    if (!p) return false;
    std::error_code ec;
    const std::filesystem::file_status st = std::filesystem::status(*p, ec);
    if (ec) {
#ifdef _WIN32
        if (ec.category() == std::system_category() &&
            (ec.value() == 21 || ec.value() == 123 || ec.value() == 1921))
            return false;
#endif
        const int e = util::errno_of(ec);
        if (e == ENOENT || e == ENOTDIR || e == EBADF || e == ELOOP) return false;
        throw OsError(e, path);
    }
    return st.type() == std::filesystem::file_type::regular;
}

// sys.stdin.isatty().
bool stdin_is_terminal() {
#ifdef _WIN32
    const HANDLE h = ::GetStdHandle(STD_INPUT_HANDLE);
    return h != nullptr && h != INVALID_HANDLE_VALUE && ::GetFileType(h) == FILE_TYPE_CHAR;
#else
    return ::isatty(0) == 1;
#endif
}

// The stop a signal asked for, and the end of the subcommand (which a
// Windows console's close waits for). Process-wide by nature: a signal
// handler has nowhere else to put it. Lock-free atomics, which a handler
// may touch.
std::atomic<bool> g_signalled{false};
std::atomic<bool> g_finished{false};

#ifdef _WIN32
BOOL WINAPI on_console(DWORD ev) {
    g_signalled.store(true);
    if (ev == CTRL_CLOSE_EVENT || ev == CTRL_LOGOFF_EVENT || ev == CTRL_SHUTDOWN_EVENT) {
        // the process ends when this returns: the stop's bound first
        for (int i = 0; i < 60 && !g_finished.load(); ++i) ::Sleep(50);
    }
    return TRUE;
}
#else
extern "C" void on_signal(int) { g_signalled.store(true); }
#endif

void install_signals() {
#ifdef _WIN32
    ::SetConsoleCtrlHandler(on_console, TRUE);
#else
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    std::signal(SIGPIPE, SIG_IGN);
#endif
}

// What escaped the session, as the traceback's last line.
std::string uncaught_line(const std::exception_ptr& e) {
    try {
        std::rethrow_exception(e);
    } catch (const PyError& x) {
        return x.cls() + ": " + x.what();
    } catch (const rawfile::FileError& x) {
        const OsError o(util::errno_of(x.code()), x.path());
        return o.cls() + ": " + o.what();
    } catch (const npz::FileError& x) {
        const OsError o(x.err, x.path);
        return o.cls() + ": " + o.what();
    } catch (const std::exception& x) {
        return qrss::live::error_text(x);
    } catch (...) {
        return "RuntimeError: unknown exception";
    }
}

int run(const std::string& prog, const std::vector<std::string>& argv, std::ostream& out, std::ostream& err) {
    const ArgParser ap = parser(prog);
    int status = 0;
    const std::optional<Args> args = ap.parse(argv, out, err, status);
    if (!args) return status;

    ql::Options lo;
    lo.frame = *args->text("frame");
    lo.just_ce = args->on("just_ce");
    lo.just_l = args->on("just_l");
    lo.refresh_s = *args->real("refresh");
    lo.state_dir = args->text("state");
    lo.store_dir = args->text("store");
    lo.no_store = args->on("no_store");
    lo.passband_dir = args->text("passband");
    lo.no_passband = args->on("no_passband");
    lo.recover_hours = *args->real("recover_hours");
    lo.model = args->text("model");
    lo.precision = *args->text("precision");
    lo.render = !args->on("no_pictures");
    lo.old_edge_noise = args->on("old_edge_noise");   // rules.apply(args); the environment's too
    lo.old_gating = args->on("old_gating");
    lo.renderer = listen_render::factory();

    // Every line goes out whole and flushed, one at a time (the session's
    // threads write them).
    auto err_mu = std::make_shared<std::mutex>();
    auto say = [&err, err_mu](const std::string& line) {
        std::lock_guard lk(*err_mu);
        err << line << "\n";
        err.flush();
    };
    const ql::Sink sink = [say](ql::LineKind, const std::string& text) { say(text); };

    // The source, made where the script makes it (after the locks and the listener).
    live::StreamSource* stdin_source = nullptr;
    if (const std::optional<std::string>& wav_arg = args->text("wav")) {
        const std::optional<std::string> start_text = args->text("start");
        const std::optional<std::string> slot_text = args->text("slot");
        const double speed = *args->real("speed");
        const std::string wav_path = pypath::pathlib_str(*wav_arg);
        lo.realtime = false;
        lo.source = [=](const ql::SourceContext& ctx) -> std::unique_ptr<live::Source> {
            // a bad time is the operator's typo, not a traceback (as qrss_receive.py);
            // the str sys.argv holds (as receive's --slot)
            double start = 0.0;
            if (start_text && !start_text->empty()) {
                try {
                    start = static_cast<double>(slot::parse_utc(pypath::fs_str(*start_text)).unix_us()) / 1e6;
                } catch (const std::invalid_argument& e) {
                    throw ql::Refusal("--start " + pytext::repr(pypath::fs_decode(*start_text)) + ": " + e.what());
                }
            } else if (slot_text && !slot_text->empty()) {
                try {
                    start = static_cast<double>(passband::slot_t0(slot::parse_slot(pypath::fs_str(*slot_text)))) -
                            12.0;
                } catch (const std::invalid_argument& e) {
                    throw ql::Refusal("--slot " + pytext::repr(pypath::fs_decode(*slot_text)) + ": " + e.what());
                }
            } else {
                throw ql::Refusal("--wav needs --start or --slot");
            }
            if (!path_is_file(wav_path)) throw ql::Refusal(wav_path + ": no such file");
            live::WavOptions wo;
            wo.speed = speed;
            wo.clock = ctx.clock;
            wo.sleep = ctx.sleep;
            wo.shown_as = wav_path;
            auto src = std::make_unique<live::WavSource>(wav_path, start, wo);
            for (const wav::Warning& w : src->warnings())
                say(std::string(w.kind == wav::Warning::Kind::wav_file ? "WavFileWarning: " : "RuntimeWarning: ") +
                    w.text);
            return src;
        };
    } else {
        const std::string fmt = *args->text("format");
        lo.realtime = true;
        lo.source = [fmt, &stdin_source](const ql::SourceContext& ctx) -> std::unique_ptr<live::Source> {
            if (stdin_is_terminal()) throw ql::Refusal("pipe 8 kHz mono audio into stdin (see --help), or use --wav");
            live::StreamOptions so;
            so.fmt = fmt;
            so.clock = ctx.clock;
            auto src = std::make_unique<live::StreamSource>(live::InputHandle::standard_input(), so);
            stdin_source = src.get();
            return src;
        };
    }

    install_signals();
    std::unique_ptr<ql::Session> session;
    try {
        session = ql::Session::start(std::move(lo), sink);
    } catch (const ql::Refusal& e) {
        return system_exit(err, e.what());
    } catch (const wav::ReadError& e) {
        return uncaught(err, OsError(e.err, e.path));
    } catch (const wav::FormatError& e) {
        return uncaught(err, wav_error_class(e.kind), e.what());
    } catch (const std::filesystem::filesystem_error& e) {
        return uncaught(err, OsError(util::errno_of(e.code()), util::path_text(e.path1())));
    } catch (...) {
        err << uncaught_line(std::current_exception()) << "\n";
        return 1;
    }

    // until the end of the audio, or a signal
    while (!session->wait_ended(0.05))
        if (g_signalled.load()) session->interrupt();   // returns once ended

    int rc = 0;
    if (const std::exception_ptr e = session->error()) {
        say(uncaught_line(e));
        rc = 1;
    } else if (stdin_source) {
        if (const std::exception_ptr e = stdin_source->error()) {
            say(uncaught_line(e));
            rc = 1;
        }
    }
    out.flush();
    err.flush();
    g_finished.store(true);
    if (session->abandoned()) {
        // threads still at an abandoned receive: no static destructors under them
        std::fflush(nullptr);
        std::_Exit(rc);
    }
    session.reset();
    return rc;
}

const Subcommand::Register reg("listen", "qrss_listen.py",
                               "listen live on stdin (or a replayed WAV), keeping the QRSS window's tiles", &run);

}  // namespace
}  // namespace sstvae::qrss::cli
