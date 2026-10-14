// sstvae-qrss decode: qrss_decode.py. A QRSSTVAE picture from the
// multi-pass store (or one stored pass) to a PNG, or the store's list.
//
// The script, step by step, with how it ends at each one:
//   parse_args                                     argparse's refusals (exit 2)
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
//         --retro, --em N > 0                     refused here (below)
//         acc_lines(acc, store)
//   render.load(precision, model, codec_id, force=--any-codec, warn=print)
//         CodecMismatch, resolve_onnx's refusals  SystemExit; onnxruntime's: uncaught
//   render.render(...).save(OUT)                  OSError on open
//   "wrote <OUT>"
// (core/qrss/decode.hpp has parse_key, acc_lines, member_wave and
// planes_from_pass; decode.hpp the picture hook.)
//
// What this port does not do, and where it says so:
// - --em and --retro (leave-one-out EM, the passband store's search) are
//   not ported yet: on the --key path, where the script would start
//   either, this refuses as argparse refuses ("<prog>: error: --em and
//   --retro are not in this build yet; use qrss_decode.py", exit 2).
//   Everything before is the script's, refusals included; on --pass and
//   --list the script ignores both, and so does this.
// - The picture is PNG only (Pillow picks the format by the extension),
//   refused at the render step otherwise; without the codec compiled in
//   (codec/decode_image.cpp) every picture is refused there, after the
//   lines the script prints first. The PNG's pixels are the reference's,
//   its bytes stb's encoder's.
// - A callsign holding '.' (the beacon alphabet has it): store.py names
//   the accumulator's files with Path.with_suffix, which cuts the name at
//   its last '.' ("K1.AB_00000001" -> acc/K1.json), where the C++ store
//   appends (acc/K1.AB_00000001.json). Such a picture in a store the
//   other implementation wrote is "no picture" on --key (and --list ends
//   in the store's KeyError); store.py is the one to fix, since two
//   callsigns differing after their last '.' share its file.
//
// --key's picture ID is any hexadecimal int, as int() reads it: one past
// int64 keys no accumulator the store can hold, so it is "no picture", the
// name the script would look for being spelled in full (an OSError for a
// name too long to stat, as there).

#include <cerrno>
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
#include "qrss/npz.hpp"
#include "qrss/pypath.hpp"
#include "qrss/pytext.hpp"
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
    "    sstvae-qrss decode out.png --pass ~/.local/share/qrsstvae/passes/<uid>.npz\n"
    "    sstvae-qrss decode --list\n"
    "\n"
    "--key CALL:PID renders the accumulator of that callsign and picture ID\n"
    "(hexadecimal) in the store (--store DIR, default $QRSSTVAE_HOME or\n"
    "~/.local/share/qrsstvae). --pass renders a single stored pass, which needs a\n"
    "decoded header to place its latents. --list prints the store's\n"
    "accumulators and exits.\n"
    "\n"
    "qrss_decode.py's --em N (leave-one-out EM over the accumulator's passes)\n"
    "and --retro (a search of the 48 h passband store for passes too weak to\n"
    "detect when they arrived) are not ported to sstvae-qrss yet: with --key\n"
    "they are refused, and --passband and --frame (which serve --retro) and\n"
    "--old-edge-noise (which serves both) change nothing. Use qrss_decode.py\n"
    "for them.\n"
    "\n"
    "An accumulator may hold passes of either waveform, CE and L, and they\n"
    "decode alike (latents in air order).\n"
    "\n"
    "Prints, per segment of the picture, the effective SNR (10 log10 median\n"
    "W) and the members with each one's waveform and frame.\n"
    "The codec is the published v5 decoder (--model DIR, --precision). The\n"
    "picture's codec ID (from the accumulator or the pass's header) must\n"
    "match the decoder's, or nothing is decoded (--any-codec overrides).\n"
    "The picture is written as PNG only (a .png name).";

constexpr const char* EM_RETRO_REFUSAL = "--em and --retro are not in this build yet; use qrss_decode.py";

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

// Whether Path(path).stat() finds something, as Python 3.13's Path.exists /
// Path.is_file decide: false for the errors pathlib reads as "not there"
// (ENOENT, ENOTDIR, EBADF, ELOOP, and on Windows ERROR_NOT_READY,
// ERROR_INVALID_NAME, ERROR_CANT_RESOLVE_FILENAME), the OSError for any
// other ("File name too long"). `regular`: is_file rather than exists.
bool pathlib_stat(const std::string& path, bool regular) {
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
    return regular ? st.type() == std::filesystem::file_type::regular
                   : st.type() != std::filesystem::file_type::not_found;
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
    // rules.apply(args): --old-edge-noise sets a receiver rule that only
    // --em and --retro (refused below) would use.

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
            if (!pathlib_stat(path, true)) return system_exit(err, "--pass " + path + ": no such file");
            const types::PassResult p = load_pass(path);
            planes = dq::planes_from_pass(p);
            codec_id = p.header->codec_id;
            out << dq::pass_line(p) << "\n";
        } else {
            store::Store st(args->text("store"));
            const dq::PictureKey key = dq::parse_key(*key_text);
            if (!pathlib_stat(acc_index_path(st, key), false))
                return system_exit(err, "no picture " + key.callsign + " " + dq::py_hex(key.picture_id.decimal, 8) +
                                            " in " + st.root());
            if (retro || (em.decimal != "0" && em.decimal.front() != '-'))
                return usage_error(ap, prog, err, EM_RETRO_REFUSAL);
            if (!key.picture_id.value)   // a file named for an ID no store key can hold
                return system_exit(err, "--key: picture ID " + key.picture_id.decimal +
                                            " is past the 64-bit IDs sstvae-qrss's store keys; use qrss_decode.py");
            const store::Accumulator acc = st.accumulator(store::Key{key.callsign, *key.picture_id.value});
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
                               "decode a picture from the multi-pass store or one pass to a PNG "
                               "(--em and --retro not yet)",
                               &run);

}  // namespace
}  // namespace sstvae::qrss::cli
