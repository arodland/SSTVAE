// sstvae-qrss encode: qrss_encode.py. Encodes an image once into a stored
// picture (.qrsp); every pass of it sends those fp16 latents.
//
// Built only with the codec (sstvae_qrss_codec). The script, step by step,
// with how it ends at each one:
//   os.path.isfile(image) or SystemExit("<image>: no such file")
//   --codec-id: int(text, 16) (an uncaught ValueError), then
//       SystemExit("--codec-id must be 16 bits, not <text>")
//   else: resolve_onnx("encoder", --model, --precision) (its SystemExits),
//       codec_id_from_onnx(enc) (onnxruntime's exception, uncaught), None ->
//       SystemExit("<enc> carries no sstvae.source_sha256 metadata; ...")
//   load_codec(--model, precision=...): a .pt model is the torch backend,
//       which refuses any precision: an uncaught ValueError
//   load_image(image): OSError/ValueError -> SystemExit("<image>: <error>")
//   codec.encode: the encoder part resolved now (resolve_onnx's SystemExits)
//   StoredPicture.from_latents, save_qrsp, print("wrote ...")
//
// With no --model the published encoder comes from the native model cache
// (fetched on first use when the build has the Qt fetcher), where the
// script uses huggingface_hub's; the two never share a cache, by design
// (CLAUDE.md, "Model artifacts").

#include <cstdint>
#include <cstdio>
#include <exception>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "argparse.hpp"
#include "checkpoint/checkpoint.hpp"
#include "common.hpp"
#include "qrss/codec/codec.hpp"
#include "qrss/picture.hpp"
#include "qrss/pynum.hpp"
#include "subcommand.hpp"

namespace sstvae::qrss::cli {
namespace {

namespace pc = sstvae::qrss::picture;
namespace qc = sstvae::qrss::codec;

constexpr const char* DESCRIPTION =
    "Encode an image once into a stored QRSSTVAE picture (.qrsp).\n"
    "\n"
    "    sstvae-qrss encode photo.jpg pic.qrsp --mode A\n"
    "\n"
    "The picture is encoded once and kept as the fp16 latents every pass will\n"
    "send; repeats never re-encode (a re-encode would send a different picture\n"
    "under the same ID). Prints the picture ID. The codec ID is read from the\n"
    "model's `sstvae.source_sha256` metadata; a model without it is refused\n"
    "unless --codec-id is given, because receivers must decode with the same\n"
    "checkpoint.";

// sstvae.codec.MODEL_HELP.
constexpr const char* MODEL_HELP =
    "model to use: a directory of exported .onnx files, a single .onnx (its sibling part is found beside "
    "it), or a .pt checkpoint (needs torch, which the app extras do not install). Defaults to the published "
    "ONNX artifacts, downloaded and cached on first use";

ArgParser parser(const std::string& prog) {
    std::vector<std::string> precisions(checkpoint::PRECISIONS.begin(), checkpoint::PRECISIONS.end());
    std::vector<std::string> modes(pc::MODE_NAMES.begin(), pc::MODE_NAMES.end());
    ArgParser ap(prog, DESCRIPTION);
    ap.positional("image");
    ap.positional("output", "stored picture (.qrsp)");
    ap.option("--model", ArgSpec().described(MODEL_HELP));
    ap.option("--precision", ArgSpec().one_of(precisions).otherwise(std::string(checkpoint::DEFAULT_PRECISION)));
    ap.option("--mode", ArgSpec().one_of(modes).otherwise("A").described(
                            "A sends group 0, B groups 0-1, C groups 0-2 (one per slot)"));
    ap.option("--codec-id",
              ArgSpec().shown_as("HEX").described("16-bit codec ID, for a model without source metadata"));
    return ap;
}

std::string hex4(std::int64_t v) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%04llx", static_cast<unsigned long long>(v));
    return buf;
}

int run(const std::string& prog, const std::vector<std::string>& argv, std::ostream& out, std::ostream& err) {
    int status = 0;
    const std::optional<Args> args = parser(prog).parse(argv, out, err, status);
    if (!args) return status;
    const std::string image = *args->text("image");
    const std::string output = *args->text("output");
    const std::optional<std::string> model = args->text("model");
    const std::string precision = *args->text("precision");
    const std::string mode_name = *args->text("mode");
    const std::optional<std::string> codec_id_text = args->text("codec_id");

    if (!is_file(image)) return system_exit(err, image + ": no such file");

    std::int64_t codec_id = 0;
    if (codec_id_text) {
        const std::optional<pynum::PyInt> v = pynum::py_int(*codec_id_text, 16);
        if (!v) return uncaught(err, "ValueError", pynum::invalid_int_literal(*codec_id_text, 16));
        if (!v->value || *v->value < 0 || *v->value >= (1 << 16))
            return system_exit(err, "--codec-id must be 16 bits, not " + *codec_id_text);
        codec_id = *v->value;
    } else {
        std::string enc;
        std::optional<std::int64_t> found;
        try {
            enc = qc::resolve_onnx("encoder", model, precision);
            found = qc::codec_id_from_onnx(enc, precision);
        } catch (const checkpoint::CheckpointError& e) {
            return system_exit(err, e.what());
        } catch (const qc::OrtError& e) {
            return uncaught(err, "onnxruntime.capi.onnxruntime_pybind11_state." + e.python_class(), e.what());
        }
        if (!found)
            return system_exit(err, enc + " carries no sstvae.source_sha256 metadata; pass --codec-id HEX so "
                                          "receivers know which decoder to use");
        codec_id = *found;
    }

    // load_codec(model, precision=...) sends a .pt to the torch backend,
    // which refuses a precision -- and the script always passes one.
    if (model && qc::pathlib_suffix(*model) == ".pt")
        return uncaught(err, "ValueError", "precision applies to the ONNX backend only");

    images::ImageArray img;
    try {
        img = qc::load_image(image);
    } catch (const std::exception& e) {
        return system_exit(err, image + ": " + e.what());
    }

    std::int64_t mode = 0;
    while (pc::MODE_NAMES[static_cast<std::size_t>(mode)] != mode_name) ++mode;

    pc::StoredPicture sp;
    try {
        sp = qc::encode_picture(img, mode, codec_id, qc::onnx_encoder(model, precision));
    } catch (const checkpoint::CheckpointError& e) {
        return system_exit(err, e.what());
    } catch (const qc::OrtError& e) {
        return uncaught(err, "onnxruntime.capi.onnxruntime_pybind11_state." + e.python_class(), e.what());
    } catch (const std::invalid_argument& e) {
        return uncaught(err, "ValueError", e.what());
    } catch (const std::exception& e) {
        return uncaught(err, "RuntimeError", e.what());
    }

    try {
        write_file(output, pc::qrsp_bytes(sp));
    } catch (const PyError& e) {
        return uncaught(err, e);
    }
    out << "wrote " << output << ": mode " << mode_name << " (" << mode + 1 << " segment" << (mode ? "s" : "")
        << "), codec " << hex4(codec_id) << ", picture ID " << sp.id_hex() << "\n";
    return 0;
}

const Subcommand::Register reg("encode", "qrss_encode.py", "encode an image once into a stored picture (.qrsp)",
                               &run);

}  // namespace
}  // namespace sstvae::qrss::cli
