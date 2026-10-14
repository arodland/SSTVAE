// sstvae-qrss decode: qrss_decode.py. A QRSSTVAE picture from the
// multi-pass store (or one stored pass) to a PNG, or the store's list.
//
// The script, step by step, with how it ends at each one:
//   parse_args                                     argparse's refusals (exit 2)
//   rules.apply(args)                              --old-edge-noise sets QRSS_OLD_EDGE_NOISE
//   --list: Store(--store)                         OSError if its directories cannot be made
//         "no pictures in the store" when there are none; else acc_lines(acc, store)
//         for every accumulator, sorted by file name
//   no OUT.png, or neither --key nor --pass        ap.error("give OUT.png and one of --key or
//                                                  --pass (or --list)"), exit 2
//   --pass: "--em and --retro need an accumulator (--key); ignored for a single pass"
//         when either was given; Path.is_file()   SystemExit("--pass <path>: no such file")
//         PassResult.load                         uncaught (np.load's, KeyError, ...)
//         planes_from_pass                        SystemExit without a decoded header
//         "pass <uid>: <wave> <frame> frame, mean W <dB> dB"
//   --key: Store(--store), parse_key               SystemExit("--key '<s>': ...")
//         Path.exists() of its index              SystemExit("no picture <CALL> <pid> in <root>")
//         --retro: Path.is_dir() of --passband    SystemExit("--retro: no passband store at <dir>")
//           (default STORE/passband); --frame     SystemExit("--frame 'l-short': --retro searches
//                                                  for CE passes only" / "... not one of full,
//                                                  medium, short, tiny")
//           em.retro_detect, its lines prefixed "retro: ", then "retro: <n> pass(es) found"
//         --em N > 0: em.em_refine, its lines, then "EM: mean W <a> -> ... -> <b> dB (no
//           change)" or "... (+x.xx dB)", and "; W is the receiver's own estimate, ..."
//         acc_lines(acc, store)
//   render.load(precision, model, codec_id, force=--any-codec, warn=print)
//         CodecMismatch, resolve_onnx's refusals  SystemExit; onnxruntime's: uncaught
//   render.render(...).save(OUT)                  OSError on open
//   "wrote <OUT>"
// (core/qrss/decode.hpp has parse_key, acc_lines, member_wave and
// planes_from_pass; core/qrss/em_api.hpp the EM and retro calls, which are
// em_refine.hpp's and em_search.hpp's ports of em.py; decode.hpp the
// picture hook.)
//
// --em and --retro are the C++ ports: what they write to the store and the
// lines they print are the script's, numbers to the ports' bars
// (em_refine.hpp, em_search.hpp), refusals and exit codes exactly. A store
// error while they run ends as the script's does (the store's KeyError,
// an OSError). --em ROUNDS past the 64-bit integers, or past 2^31, runs as
// 2^31 rounds (em_api::MAX_ROUNDS): the rounds stop at EM_TOL_DB long
// before either, as the script's would.
//
// What this port does differently, and where it says so:
// - The picture is PNG only (Pillow picks the format by the extension),
//   refused at the render step otherwise; without the codec compiled in
//   (codec/decode_image.cpp) every picture is refused there, after the
//   lines the script prints first (EM and retro included). The PNG's
//   pixels are the reference's, its bytes stb's encoder's.
// - A callsign holding '.' (the beacon alphabet has it): store.py names
//   the accumulator's files with Path.with_suffix, which cuts the name at
//   its last '.' ("K1.AB_00000001" -> acc/K1.json), where the C++ store
//   appends (acc/K1.AB_00000001.json). Such a picture in a store the
//   other implementation wrote is "no picture" on --key (and --list ends
//   in the store's KeyError); store.py is the one to fix, since two
//   callsigns differing after their last '.' share its file.
// - A damaged store (a truncated or corrupt pass or accumulator file, or
//   --retro's passband meta.json) ends both, but not with the same
//   exception: the script's tracebacks name np.load's and json's classes
//   (BadZipFile, EOFError, JSONDecodeError, ...), the C++ stores throw
//   their ValueError with their own text. And in member_wave (--list, --key) the script catches only
//   OSError, KeyError and ValueError, so a pass file np.load cannot open
//   (BadZipFile, EOFError) crashes the listing part-way, where this prints
//   "?" for that member and goes on; the script is the one to fix.
//
// --key's picture ID is any hexadecimal int, as int() reads it: one past
// int64 keys no accumulator the store can hold, so it is "no picture", the
// name the script would look for being spelled in full (an OSError for a
// name too long to stat, as there).

#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include "argparse.hpp"
#include "common.hpp"
#include "decode.hpp"
#include "math/errors.hpp"
#include "qrss/decode.hpp"
#include "qrss/em.hpp"
#include "qrss/em_api.hpp"
#include "qrss/frame.hpp"
#include "qrss/npz.hpp"
#include "qrss/pypath.hpp"
#include "qrss/pytext.hpp"
#include "qrss/rules.hpp"
#include "qrss/store.hpp"
#include "qrss/types.hpp"
#include "subcommand.hpp"
#include "util/fileio.hpp"

namespace sstvae::qrss::cli {

namespace decode_image {

namespace {
PictureRenderer& renderer_slot() {
    static PictureRenderer r;   // set during static initialisation, read-only after
    return r;
}
}  // namespace

PictureRenderer picture_renderer() { return renderer_slot(); }

RegisterPictureRenderer::RegisterPictureRenderer(PictureRenderer r) { renderer_slot() = r; }

}  // namespace decode_image

namespace {

namespace dq = sstvae::qrss::decode;

// qrss_decode.py's docstring, its --help description, as this port has it.
constexpr const char* DESCRIPTION =
    "Decode a QRSSTVAE picture from the multi-pass store (or one pass) to a PNG.\n"
    "\n"
    "    sstvae-qrss decode out.png --key K1ABC:1a2b3c4d\n"
    "    sstvae-qrss decode out.png --key K1ABC:1a2b3c4d --em 3\n"
    "    sstvae-qrss decode out.png --key K1ABC:1a2b3c4d --retro\n"
    "    sstvae-qrss decode out.png --pass ~/.local/share/qrsstvae/passes/<uid>.npz\n"
    "    sstvae-qrss decode --list\n"
    "\n"
    "--key CALL:PID renders the accumulator of that callsign and picture ID\n"
    "(hexadecimal) in the store (--store DIR, default $QRSSTVAE_HOME or\n"
    "~/.local/share/qrsstvae). --em N first runs N rounds of leave-one-out EM\n"
    "over its passes (each pass re-tracked from its stored capture with the\n"
    "others as its reference) and stores the improved passes. --pass renders a\n"
    "single stored pass, which needs a decoded header to place its latents.\n"
    "--list prints the store's accumulators and exits.\n"
    "\n"
    "--retro first searches the 48 h passband store (--passband DIR, default\n"
    "STORE/passband, kept by `sstvae-qrss listen`) for passes of the picture\n"
    "too weak to detect when they arrived: every stored slot without a member\n"
    "is template-searched with the accumulator's soft data (--frame, default\n"
    "full), and what is found is received and associated.\n"
    "\n"
    "An accumulator may hold passes of either waveform, CE and L, and they\n"
    "decode alike (latents in air order). EM and --retro are CE-only: an L\n"
    "member is left as it was received and says so, and --retro searches for\n"
    "CE passes (an \"l-...\" --frame is refused).\n"
    "\n"
    "Prints, per segment of the picture, the effective SNR (10 log10 median\n"
    "W) and the members with each one's waveform and frame, then the mean W\n"
    "before and after each EM round.\n"
    "The codec is the published v5 decoder (--model DIR, --precision). The\n"
    "picture's codec ID (from the accumulator or the pass's header) must\n"
    "match the decoder's, or nothing is decoded (--any-codec overrides).\n"
    "The picture is written as PNG only (a .png name).";

ArgParser parser(const std::string& prog) {
    ArgParser ap(prog, DESCRIPTION);
    ap.optional_positional("output", "PNG to write");
    ap.option("--key", ArgSpec().described("CALL:PICTUREID (hex) of an accumulator"));
    // dest="pass_file": the metavar argparse derives from it
    ap.option("--pass", ArgSpec().shown_as("PASS_FILE").described("a single stored pass (.npz)"));
    ap.flag("--list", "list the store's pictures");
    ap.exclusive({"--key", "--pass", "--list"});
    ap.option("--store", ArgSpec().described("multi-pass store directory"));
    ap.option("--em", ArgSpec().integer().otherwise("0").shown_as("ROUNDS").described(
                          "leave-one-out EM rounds before decoding (default 0)"));
    ap.flag("--retro", "search the passband store for weaker passes first (--key)");
    ap.option("--passband", ArgSpec().described("passband store directory (default STORE/passband)"));
    ap.option("--frame", ArgSpec().otherwise("full").described("frame shape for --retro (default full)"));
    ap.flag("--any-codec", "decode even when the picture's codec ID is not the decoder's");
    ap.option("--model", ArgSpec().described("codec model directory"));
    ap.option("--precision", ArgSpec().one_of({"fp32", "fp16", "int8"}).otherwise("fp32").described(
                                 "codec precision"));
    // rules.add_arguments(ap, (rules.EDGE_NOISE,))
    ap.flag("--old-edge-noise", "take the noise floor near 300 and 2700 Hz as before 2026-10-12, partly off "
                                "the front end's filter skirt");
    return ap;
}

// ap.error(message): the usage and "<prog>: error: <message>" on stderr, exit 2.
int usage_error(const ArgParser& ap, const std::string& prog, std::ostream& err, const std::string& message) {
    err << ap.usage() << "\n" << prog << ": error: " << message << "\n";
    return 2;
}

// What Path(path).stat() is asked: exists, is_file or is_dir.
enum class Want { exists, file, dir };

// Whether Path(path).stat() finds `want`, as Python 3.13's Path.exists /
// Path.is_file / Path.is_dir decide: false for the errors pathlib reads as
// "not there" (ENOENT, ENOTDIR, EBADF, ELOOP, and on Windows
// ERROR_NOT_READY, ERROR_INVALID_NAME, ERROR_CANT_RESOLVE_FILENAME), the
// OSError for any other ("File name too long").
bool pathlib_stat(const std::string& path, Want want) {
    const auto p = util::fs_path(path);
    if (!p) return false;   // a name the OS cannot spell: ValueError, read as False
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
    switch (want) {
        case Want::file: return st.type() == std::filesystem::file_type::regular;
        case Want::dir: return st.type() == std::filesystem::file_type::directory;
        case Want::exists: break;
    }
    return st.type() != std::filesystem::file_type::not_found;
}

// Store._acc_paths(key)[0] as the C++ store spells it (store.hpp), or, for
// a picture ID past int64, the name the reference would look for:
// root/acc/<CALL, '/' as '_'>_<pid:08x>.json.
std::string acc_index_path(const store::Store& st, const dq::PictureKey& key) {
    if (key.picture_id.value) return st.acc_paths(store::Key{key.callsign, *key.picture_id.value}).first;
    std::string call = key.callsign;
    for (char& c : call)
        if (c == '/') c = '_';
    return pypath::child(pypath::child(st.root(), "acc"), call + "_" + dq::py_hex(key.picture_id.decimal, 8) + ".json");
}

// PassResult.load(path) as the script, which does not catch, ends with it:
// np.load's decision on the first bytes (EOFError for an empty file, the
// pickle refusal for what is no zip, a TypeError for an .npy), then the
// archive (zipfile.BadZipFile, numpy's ValueError for a member's header)
// and the record's own refusals (KeyError, ValueError, TypeError).
types::PassResult load_pass(const std::string& path) {
    const std::vector<std::uint8_t> data = read_file(path);
    auto starts = [&](std::string_view magic) {
        if (data.size() < magic.size()) return false;
        for (std::size_t k = 0; k < magic.size(); ++k)
            if (static_cast<std::uint8_t>(magic[k]) != data[k]) return false;
        return true;
    };
    if (data.empty()) throw PyError("EOFError", "No data left in file");
    if (!starts(std::string_view("PK\x03\x04", 4)) && !starts(std::string_view("PK\x05\x06", 4))) {
        if (starts(std::string_view("\x93NUMPY", 6)))
            throw PyError("TypeError", "'numpy.ndarray' object does not support the context manager protocol");
        throw PyError("ValueError",
                      "This file contains pickled (object) data. If you trust the file you can load it unsafely "
                      "using the `allow_pickle=` keyword argument or `pickle.load()`.");
    }
    npz::Npz z;
    try {
        z = npz::parse_npz(data);
    } catch (const std::exception& e) {
        const std::string what = e.what();
        throw PyError(what.rfind("npy: ", 0) == 0 ? "ValueError" : "zipfile.BadZipFile", what);
    }
    try {
        return types::PassResult::from_npz(z);
    } catch (const store::KeyError&) {
        throw;   // the record's own: a key its meta lacks
    } catch (const std::out_of_range& e) {
        // NpzFile[name] of a member the archive lacks: a KeyError of the text
        throw PyError("KeyError", pytext::repr(std::string_view(e.what())));
    }
}

int run(const std::string& prog, const std::vector<std::string>& argv, std::ostream& out, std::ostream& err) {
    const ArgParser ap = parser(prog);
    int status = 0;
    const std::optional<Args> args = ap.parse(argv, out, err, status);
    if (!args) return status;
    // rules.apply(args): --old-edge-noise sets the receiver rule (in the
    // environment, where the receiver under --em and --retro reads it).
    rules::apply(args->on("old_edge_noise"), false);

    try {
        if (args->on("list")) {
            store::Store st(args->text("store"));
            const std::vector<store::Key> keys = st.open_keys();
            if (keys.empty()) out << "no pictures in the store\n";
            for (const store::Key& key : keys)
                for (const std::string& line : dq::acc_lines(st.accumulator(key), &st)) out << line << "\n";
            return 0;
        }
        const std::optional<std::string>& output = args->text("output");
        const std::optional<std::string>& key_text = args->text("key");
        const std::optional<std::string>& pass_text = args->text("pass");
        if (!output || (!key_text && !pass_text))
            return usage_error(ap, prog, err, "give OUT.png and one of --key or --pass (or --list)");

        const pynum::PyInt em = *args->integer("em");
        const bool retro = args->on("retro");
        dq::Planes planes;
        std::int64_t codec_id = 0;
        if (pass_text) {
            if (em.decimal != "0" || retro)
                out << "--em and --retro need an accumulator (--key); ignored for a single pass\n";
            const std::string path = pypath::pathlib_str(*pass_text);
            if (!pathlib_stat(path, Want::file)) return system_exit(err, "--pass " + path + ": no such file");
            const types::PassResult p = load_pass(path);
            planes = dq::planes_from_pass(p);
            codec_id = p.header->codec_id;
            out << dq::pass_line(p) << "\n";
        } else {
            store::Store st(args->text("store"));
            const dq::PictureKey key = dq::parse_key(*key_text);
            if (!pathlib_stat(acc_index_path(st, key), Want::exists))
                return system_exit(err, "no picture " + key.callsign + " " + dq::py_hex(key.picture_id.decimal, 8) +
                                            " in " + st.root());
            if (!key.picture_id.value)   // a file named for an ID no store key can hold
                return system_exit(err, "--key: picture ID " + key.picture_id.decimal +
                                            " is past the 64-bit IDs sstvae-qrss's store keys; use qrss_decode.py");
            const store::Key skey{key.callsign, *key.picture_id.value};
            if (retro) {
                // Path(args.passband) if args.passband else store.root / "passband"
                const std::optional<std::string>& pb_text = args->text("passband");
                const std::string pb_dir = pb_text && !pb_text->empty() ? pypath::pathlib_str(*pb_text)
                                                                        : em_api::default_passband_dir(st);
                if (!pathlib_stat(pb_dir, Want::dir))
                    return system_exit(err, "--retro: no passband store at " + pb_dir);
                try {
                    em_api::retro_frame(*args->text("frame"));
                } catch (const em_api::Refusal& e) {
                    return system_exit(err, e.what());
                }
                em::Hooks hooks;
                hooks.log = [&out](const std::string& line) { out << "retro: " << line << "\n" << std::flush; };
                const em_api::RetroResult r =
                    em_api::retro(st, skey, em_api::Passband::at(pb_dir), *args->text("frame"), {}, hooks);
                const std::size_t n = r.finds.size();
                out << "retro: " << n << " pass" << (n != 1 ? "es" : "") << " found\n";
            }
            // args.em > 0 (a value past int64 is positive unless it is negative)
            if (em.value ? *em.value > 0 : em.decimal.front() != '-') {
                em_api::RefineOptions o;
                o.rounds = em.value ? *em.value : em_api::MAX_ROUNDS;
                em::Hooks hooks;
                hooks.log = [&out](const std::string& line) { out << line << "\n" << std::flush; };
                const std::vector<double> hist = em_api::refine(st, skey, o, hooks).history;
                const double gain = hist.back() - hist.front();
                std::string line = "EM: mean W ";
                for (std::size_t k = 0; k < hist.size(); ++k)
                    line += (k ? " -> " : "") + frame::py_format(hist[k], "%+.2f");
                line += " dB";
                // abs(gain) < 0.005: false for a NaN gain (-inf to -inf), printed "+nan"
                line += std::fabs(gain) < 0.005 ? " (no change)" : " (" + frame::py_format(gain, "%+.2f") + " dB)";
                out << line << "; W is the receiver's own estimate, and a re-receive can raise it by more than "
                               "the picture gains\n";
            }
            const store::Accumulator acc = st.accumulator(skey);
            for (const std::string& line : dq::acc_lines(acc, &st)) out << line << "\n";
            planes = dq::planes_from_accumulator(acc);
            codec_id = acc.codec_id;
        }

        const decode_image::PictureRenderer r = decode_image::picture_renderer();
        const std::string shown = pypath::pathlib_str(*output);
        if (!r.render)
            return system_exit(err, shown + ": this sstvae-qrss renders no pictures (the codec half of decode is not "
                                            "compiled in: it needs SSTVAE_BUILD_CODEC and "
                                            "apps/sstvae_qrss/codec/decode_image.cpp); use qrss_decode.py");
        // render.model_dir(): $QRSSTVAE_MODEL_DIR when --model is absent.
        std::optional<std::string> model = args->text("model");
        if (!model) {
            const char* env = std::getenv("QRSSTVAE_MODEL_DIR");
            if (env && *env) model = env;
        }
        const decode_image::PictureRequest req{*output, model, *args->text("precision"), args->on("any_codec")};
        if (r.refusal)
            if (const std::optional<std::string> why = r.refusal(req)) return system_exit(err, *why);
        return r.render(planes, codec_id, req, out, err);
    } catch (const dq::Refusal& e) {
        return system_exit(err, e.what());
    } catch (const em_api::Refusal& e) {   // the API's own checks (a picture gone since the one above)
        return system_exit(err, e.what());
    } catch (const dq::PyException& e) {
        return uncaught(err, e.cls(), e.what());
    } catch (const PyError& e) {
        return uncaught(err, e);
    } catch (const npz::FileError& e) {
        return uncaught(err, OsError(e.err, e.path));
    } catch (const store::KeyError& e) {
        err << e.what() << "\n";   // "KeyError: '<key>'"
        return 1;
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

const Subcommand::Register reg("decode", "qrss_decode.py",
                               "decode a picture from the multi-pass store or one pass to a PNG",
                               &run);

}  // namespace
}  // namespace sstvae::qrss::cli
