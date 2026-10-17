// native/core/qrss/codec (the codec glue) against the reference's
// behaviour: picture.codec_id_from_onnx, checkpoint.resolve_onnx with an
// explicit path, pathlib's spelling of a path, and the encode seam.
//
// Built only with SSTVAE's codec (onnxruntime). The models whose metadata
// is read are written by this test: an Identity graph a few hundred bytes
// long (tiny_onnx below, the same bytes as tests/qrss_codec_fixtures.py
// writes), so nothing here needs a downloaded artifact. When the v5 models
// are present ($QRSSTVAE_MODEL_DIR, else /home/user/qrss-data/models) their
// codec ID is checked too: 0xD1D8.
//
// Expected values are the reference's (CPython 3.13 and onnxruntime):
//   str(PurePosixPath(t)), PurePosixPath(t).suffix
//   int(sha[:4], 16) or None, as codec_id_from_onnx reads a source hash
// tests/test_qrss_native_codec.py compares the two implementations
// directly, the full encode through the v5 encoder included.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include "check.hpp"
#include "images/images.hpp"
#include "qrss/codec/codec.hpp"
#include "qrss/codec/encode.hpp"
#include "qrss/picture.hpp"

using namespace sstvae;
namespace qc = sstvae::qrss::codec;
namespace pc = sstvae::qrss::picture;
namespace fs = std::filesystem;

namespace {

// --- a model to read ---------------------------------------------------------

std::string varint(std::uint64_t n) {
    std::string out;
    for (;;) {
        const auto b = static_cast<char>(n & 0x7F);
        n >>= 7;
        if (!n) return out + b;
        out += static_cast<char>(b | 0x80);
    }
}

std::string field(int num, std::uint64_t value) { return varint(static_cast<std::uint64_t>(num) << 3) + varint(value); }

std::string field(int num, const std::string& payload) {
    return varint(static_cast<std::uint64_t>(num) << 3 | 2) + varint(payload.size()) + payload;
}

// y = Identity(x), float[1], ir_version 7, opset 13; `sha` in its
// metadata_props as sstvae.source_sha256 (none for nullopt).
std::string tiny_onnx(const std::optional<std::string>& sha) {
    const std::string dim = field(1, field(1, 1));
    const std::string ttype = field(1, field(1, 1) + field(2, dim));
    auto value_info = [&](const std::string& name) { return field(1, name) + field(2, ttype); };
    const std::string node = field(1, std::string("x")) + field(2, std::string("y")) + field(4, std::string("Identity"));
    const std::string graph = field(1, node) + field(2, std::string("g")) + field(11, value_info("x")) +
                              field(12, value_info("y"));
    std::string model = field(1, 7) + field(7, graph) + field(8, field(2, 13));
    if (sha) model += field(14, field(1, std::string("sstvae.source_sha256")) + field(2, *sha));
    return model;
}

void write(const fs::path& p, const std::string& bytes) {
    fs::create_directories(p.parent_path());
    std::ofstream out(p, std::ios::binary);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!out) throw std::runtime_error("cannot write " + p.string());
}

// A fresh directory under the system temp dir, removed at exit.
struct TempDir {
    fs::path path;
    TempDir() {
        const fs::path base = fs::temp_directory_path();
        for (int k = 0;; ++k) {
            path = base / ("qrss-codec-test-" + std::to_string(std::rand()) + "-" + std::to_string(k));
            std::error_code ec;
            if (fs::create_directory(path, ec)) return;
            if (k > 100) throw std::runtime_error("no temp dir");
        }
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
    std::string operator/(const std::string& rel) const { return (path / rel).string(); }
};

template <typename E, typename F>
std::string refusal(F f) {
    try {
        f();
    } catch (const E& e) {
        return e.what();
    }
    return "(no refusal)";
}

// --- Python's spelling of a path ---------------------------------------------------

void test_pathlib() {
    struct Case {
        const char *text, *str, *suffix;
    };
    const Case cases[] = {
        {"", ".", ""},          {".", ".", ""},           {"/", "/", ""},           {"//", "//", ""},
        {"///", "/", ""},       {"//a", "//a", ""},       {"a//b/", "a/b", ""},     {"./a/./b", "a/b", ""},
        {"a/../b", "a/../b", ""}, {"/./x", "/x", ""},     {"x/.", "x", ""},         {"a.", "a.", ""},
        {".onnx", ".onnx", ""}, {"x/.a.pt", "x/.a.pt", ".pt"}, {"a..pt", "a..pt", ".pt"},
        {"foo.PT", "foo.PT", ".PT"}, {"a.pt/", "a.pt", ".pt"}, {"a.tar.gz", "a.tar.gz", ".gz"},
        {"..", "..", ""},       {"..a", "..a", ".a"},     {"m/", "m", ""},
    };
    for (const Case& c : cases) {
        check::equal(qc::pathlib_str(c.text), std::string(c.str), std::string("str(Path(") + c.text + "))");
        check::equal(qc::pathlib_suffix(c.text), std::string(c.suffix), std::string("suffix of ") + c.text);
    }
}

// --- the codec ID ---------------------------------------------------------------------

void test_codec_id_of_sha() {
    struct Case {
        const char* sha;
        std::optional<std::int64_t> id;
    };
    const Case cases[] = {
        {"d1d8", 53720}, {"D1D8ffff", 53720}, {"0x12ab", 18}, {" 1a2", 418}, {"1_23", 291},
        {"-1ab", -427},  {"+1ab", 427},       {"abc", std::nullopt}, {"", std::nullopt},
        {"zz12", std::nullopt}, {"12zz", std::nullopt}, {"1__2", std::nullopt}, {"0x_1", 1},
        {"    ", std::nullopt},
    };
    for (const Case& c : cases) {
        const auto got = qc::codec_id_of_sha(c.sha);
        check::equal(got.value_or(-99999), c.id.value_or(-99999), std::string("codec_id_of_sha('") + c.sha + "')");
    }
}

void test_codec_id_from_files(const TempDir& t) {
    const std::string full = "d1d8" + std::string(60, '0');
    write(t / "f/a-encoder-fp16.onnx", tiny_onnx(full));
    write(t / "f/none-encoder-fp16.onnx", tiny_onnx(std::nullopt));
    write(t / "f/short-encoder-fp16.onnx", tiny_onnx(std::string("abc")));
    write(t / "f/garbage-encoder-fp16.onnx", "garbage bytes, not a protobuf");

    check::equal(qc::source_sha256(t / "f/a-encoder-fp16.onnx").value_or("(none)"), full, "source_sha256");
    check::is_true(!qc::source_sha256(t / "f/none-encoder-fp16.onnx"), "no metadata");
    check::equal(qc::codec_id_from_onnx(t / "f/a-encoder-fp16.onnx").value_or(-1), std::int64_t(0xD1D8),
                 "codec ID of a file");
    check::is_true(!qc::codec_id_from_onnx(t / "f/none-encoder-fp16.onnx"), "no metadata: None");
    check::is_true(!qc::codec_id_from_onnx(t / "f/short-encoder-fp16.onnx"), "a short hash: None");
    // pathlib's spelling reaches onnxruntime and its message.
    check::equal(qc::codec_id_from_onnx(t.path.string() + "//f/./a-encoder-fp16.onnx").value_or(-1), std::int64_t(0xD1D8), "a path pathlib normalises");

    const std::string garbage = t / "f/garbage-encoder-fp16.onnx";
    try {
        qc::codec_id_from_onnx(garbage);
        check::fail("garbage model", "no refusal");
    } catch (const qc::OrtError& e) {
        check::equal(e.code(), 7, "garbage: code");
        check::equal(e.python_class(), std::string("InvalidProtobuf"), "garbage: class");
        check::equal(std::string(e.what()), "[ONNXRuntimeError] : 7 : INVALID_PROTOBUF : Load model from " +
                                                garbage + " failed:Protobuf parsing failed.",
                     "garbage: text");
    }
    const std::string missing = t / "f/missing-encoder-fp16.onnx";
    try {
        qc::codec_id_from_onnx(missing);
        check::fail("missing model", "no refusal");
    } catch (const qc::OrtError& e) {
        check::equal(e.python_class(), std::string("NoSuchFile"), "missing: class");
        check::equal(std::string(e.what()), "[ONNXRuntimeError] : 3 : NO_SUCHFILE : Load model from " + missing +
                                                " failed:Load model " + missing + " failed. File doesn't exist",
                     "missing: text");
    }
}

void test_codec_id_from_directories(const TempDir& t) {
    write(t / "both/v9-encoder-fp16.onnx", tiny_onnx(std::string("e1e1")));
    write(t / "both/v9-decoder-fp16.onnx", tiny_onnx(std::string("d0d0")));
    write(t / "both/v9-decoder-fp32.onnx", tiny_onnx(std::string("f32d")));
    write(t / "dec/v9-decoder-fp16.onnx", tiny_onnx(std::string("dec0")));
    write(t / "amb/a-encoder-fp16.onnx", tiny_onnx(std::string("aaaa")));
    write(t / "amb/b-encoder-fp16.onnx", tiny_onnx(std::string("bbbb")));
    write(t / "amb/a-decoder-fp16.onnx", tiny_onnx(std::string("abab")));
    fs::create_directories(t.path / "empty");

    check::equal(qc::codec_id_from_onnx(t / "both").value_or(-1), std::int64_t(0xE1E1), "the encoder is preferred");
    check::equal(qc::codec_id_from_onnx(t / "both", "fp32").value_or(-1), std::int64_t(0xF32D), "then the decoder");
    check::equal(qc::codec_id_from_onnx(t / "dec").value_or(-1), std::int64_t(0xDEC0), "the decoder alone");
    check::equal(qc::codec_id_from_onnx(t / "amb").value_or(-1), std::int64_t(0xABAB), "ambiguous encoders: the decoder");
    check::equal(refusal<qc::NoModelFile>([&] { qc::codec_id_from_onnx(t / "empty", "int8"); }),
                 "no *-encoder-int8.onnx or *-decoder-int8.onnx in " + (t / "empty"), "neither part");
}

// --- resolve_onnx's refusals ------------------------------------------------------------

void test_resolve_onnx(const TempDir& t) {
    const std::string both = t / "both", amb = t / "amb", empty = t / "empty", dec = t / "dec";
    using M = qc::ModelError;
    check::equal(qc::resolve_onnx("encoder", both + "/"), both + "/v9-encoder-fp16.onnx", "a directory");
    check::equal(refusal<M>([&] { qc::resolve_onnx("encoder", empty); }),
                 "no *-encoder-fp16.onnx in " + empty + "\nthat directory holds no .onnx files at all", "empty");
    check::equal(refusal<M>([&] { qc::resolve_onnx("encoder", dec, "fp32"); }),
                 "no *-encoder-fp32.onnx in " + dec + "\nfound: v9-decoder-fp16.onnx", "found");
    check::equal(refusal<M>([&] { qc::resolve_onnx("encoder", amb); }),
                 "ambiguous: 2 candidates for encoder/fp16 in " + amb + " (a-encoder-fp16.onnx, b-encoder-fp16.onnx)",
                 "ambiguous");
    check::equal(qc::resolve_onnx("decoder", both + "/v9-encoder-fp16.onnx"), both + "/v9-decoder-fp16.onnx",
                 "the sibling");
    check::equal(qc::resolve_onnx("encoder", both + "/v9-encoder-int8.onnx"), both + "/v9-encoder-int8.onnx",
                 "the part named, unchecked");
    check::equal(refusal<M>([&] { qc::resolve_onnx("decoder-grad", both + "/v9-encoder-fp16.onnx"); }),
                 std::string("need the decoder-grad as well as the encoder, but v9-decoder-grad-fp32.onnx is not next "
                             "to v9-encoder-fp16.onnx"),
                 "no sibling");
    check::equal(refusal<M>([&] { qc::resolve_onnx("encoder", both + "/model.onnx", "fp32"); }),
                 std::string("cannot tell which part model.onnx is -- expected a name like v1-encoder-fp32.onnx, or "
                             "pass the directory instead"),
                 "cannot tell");
    check::equal(refusal<M>([&] { qc::resolve_onnx("encoder", std::string("x.pt")); }),
                 std::string("--model x.pt: expected a .pt checkpoint, a .onnx artifact, or a directory containing "
                             "the exported .onnx files"),
                 "neither");
    // A ModelError is SSTVAE's CheckpointError, which the CLI catches.
    check::is_true(!refusal<checkpoint::CheckpointError>([&] { qc::resolve_onnx("encoder", empty); }).empty(),
                   "a CheckpointError");
}

// --- the encode seam ---------------------------------------------------------------------

std::vector<double> closed_form() {
    std::vector<double> v(3 * 52800);
    for (std::size_t i = 0; i < v.size(); ++i) {
        const double g = static_cast<double>(i);
        v[i] = std::sin(std::fmod(0.001 * g * g, 6.283)) + 0.3 * std::cos(g);
    }
    return v;
}

void test_encode(const TempDir& t) {
    // A 640x480 PNG, read back exactly: no resize at the codec's size.
    images::Picture pic(images::IMG_W, images::IMG_H);
    for (std::size_t i = 0; i < pic.rgb.size(); ++i) pic.rgb[i] = static_cast<std::uint8_t>((i * 7919) % 251);
    const std::string png = t / "p.png";
    images::save_png(pic, png);
    const images::ImageArray a = qc::load_image(png);
    check::equal(a.width, images::IMG_W, "width");
    check::equal(a.height, images::IMG_H, "height");
    bool same = a.chw.size() == pic.rgb.size();
    const std::size_t plane = static_cast<std::size_t>(a.width) * static_cast<std::size_t>(a.height);
    for (std::size_t c = 0; same && c < 3; ++c)
        for (std::size_t k = 0; same && k < plane; ++k)
            same = a.chw[c * plane + k] == static_cast<float>(pic.rgb[k * 3 + c]) / 255.0f;
    check::is_true(same, "load_image of a 640x480 PNG is the pixels / 255");

    // The encode is the seam's: what it returns becomes the picture
    // (test_qrss_format's frozen ID for the closed-form tensor).
    int calls = 0;
    const qc::Encoder stub = [&](const images::ImageArray& img) {
        ++calls;
        check::equal(img.chw.size(), a.chw.size(), "the encoder sees the image");
        return closed_form();
    };
    const pc::StoredPicture sp = qc::encode_picture(png, 0, 0xD1D8, stub);
    check::equal(sp.picture_id, std::int64_t(0xD349C671), "picture ID through the seam");
    check::equal(calls, 1, "one encode");

    // Mode and codec ID are refused before the image or the model is touched.
    const std::string nowhere = t / "no-such.png";
    check::equal(refusal<std::invalid_argument>([&] { qc::encode_picture(nowhere, 3, 1, stub); }),
                 std::string("mode must be 0..2 (A..C), not 3"), "mode");
    check::equal(refusal<std::invalid_argument>([&] { qc::encode_picture(nowhere, 0, 1 << 16, stub); }),
                 std::string("codec_id must fit in 16 bits, not 65536"), "codec ID");
    check::equal(calls, 1, "no encode for a refusal");
    check::is_true(refusal<std::runtime_error>([&] { qc::encode_picture(nowhere, 0, 1, stub); }) != "(no refusal)",
                   "a missing image");
}

// --- the published model, when present ----------------------------------------------------------

void test_v5() {
    const char* env = std::getenv("QRSSTVAE_MODEL_DIR");
    const std::string dir = env && *env ? env : "/home/user/qrss-data/models";
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) {
        std::printf("note: no v5 models at %s; the 0xD1D8 check is skipped\n", dir.c_str());
        return;
    }
    for (const char* prec : {"fp16", "fp32"}) {
        std::optional<std::int64_t> id;
        try {
            id = qc::codec_id_from_onnx(dir, prec);
        } catch (const qc::NoModelFile&) {
            std::printf("note: no v5 %s model in %s\n", prec, dir.c_str());
            continue;
        }
        check::equal(id.value_or(-1), std::int64_t(0xD1D8), std::string("v5 codec ID ") + prec);
    }
}

}  // namespace

int main() {
    try {
        const TempDir t;
        test_pathlib();
        test_codec_id_of_sha();
        test_codec_id_from_files(t);
        test_codec_id_from_directories(t);
        test_resolve_onnx(t);
        test_encode(t);
        test_v5();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return check::report("qrss_codec");
}
