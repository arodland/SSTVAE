// sstvae-qrss's subcommands (native/apps/sstvae_qrss), run in-process
// against what qrss_beacon.py and qrss_encode.py print and write.
//
// The beacon file is the one test_beaconfile.cpp pins: the closed-form
// mode-B picture (ID 0x73EA4EF6), callsign K1ABC/P, grid FN42, whose
// to_bytes digests came from the reference. The usage and refusal texts
// are the scripts' (argparse's, CPython 3.13), with the program named
// "sstvae-qrss <sub>" in place of "qrss_<sub>.py". encode is checked only
// up to the model (its refusals of arguments, image and --model), since a
// successful encode needs the published artifact; tests/test_qrss_native_cli.py
// runs both programs side by side, the encode included.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include "argparse.hpp"
#include "check.hpp"
#include "common.hpp"
#include "qrss/beaconfile.hpp"
#include "qrss/picture.hpp"
#include "qrss_check.hpp"
#include "subcommand.hpp"

using namespace sstvae;
namespace cli = sstvae::qrss::cli;
namespace pc = sstvae::qrss::picture;
namespace fs = std::filesystem;

namespace {

struct TempDir {
    fs::path path;
    TempDir() {
        const fs::path base = fs::temp_directory_path();
        for (int k = 0;; ++k) {
            path = base / ("qrss-cli-test-" + std::to_string(std::rand()) + "-" + std::to_string(k));
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

struct Result {
    int status;
    std::string out, err;
};

Result run(const std::string& sub, const std::vector<std::string>& args) {
    const cli::Subcommand* s = cli::Subcommand::find(sub);
    if (!s) throw std::runtime_error("no subcommand " + sub);
    std::ostringstream out, err;
    const int status = s->run("sstvae-qrss " + sub, args, out, err);
    return {status, out.str(), err.str()};
}

void expect(const Result& r, int status, const std::string& out, const std::string& err, const std::string& what) {
    check::equal(r.status, status, what + ": status");
    check::equal(r.out, out, what + ": stdout");
    check::equal(r.err, err, what + ": stderr");
}

std::vector<std::uint8_t> slurp(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

bool exists(const std::string& p) {
    std::error_code ec;
    return fs::exists(p, ec);
}

pc::StoredPicture closed_form_picture() {
    std::vector<double> v(3 * 52800);
    for (std::size_t i = 0; i < v.size(); ++i) {
        const double g = static_cast<double>(i);
        v[i] = std::sin(std::fmod(0.001 * g * g, 6.283)) + 0.3 * std::cos(g);
    }
    return pc::StoredPicture::from_latents(v, 0xD1D8, 1);
}

// --- the registry ----------------------------------------------------------------

void test_registry() {
    const auto all = cli::Subcommand::all();
    check::is_true(!all.empty(), "subcommands registered");
    for (std::size_t k = 1; k < all.size(); ++k)
        check::is_true(std::string(all[k - 1].name) < all[k].name, "sorted by name");
    const cli::Subcommand* b = cli::Subcommand::find("beacon");
    check::is_true(b != nullptr, "beacon registered");
    if (b) check::equal(std::string(b->script), std::string("qrss_beacon.py"), "beacon's script");
    check::is_true(cli::Subcommand::find("frobnicate") == nullptr, "unknown");
}

// --- argparse ------------------------------------------------------------------------

const std::string BEACON_USAGE =
    "usage: sstvae-qrss beacon [-h] --callsign CALLSIGN [--grid GRID] [--segment SEGMENT] [--ook] input output\n";

std::string usage_error(const std::string& msg) { return BEACON_USAGE + "sstvae-qrss beacon: error: " + msg + "\n"; }

void test_argparse() {
    expect(run("beacon", {}), 2, "",
           usage_error("the following arguments are required: input, output, --callsign"), "no arguments");
    expect(run("beacon", {"a", "--bogus"}), 2, "",
           usage_error("the following arguments are required: output, --callsign"), "required before unknown");
    expect(run("beacon", {"a", "b", "--callsign", "K1", "--bogus=1", "c"}), 2, "",
           usage_error("unrecognized arguments: --bogus=1 c"), "unrecognized");
    expect(run("beacon", {"a", "b", "--callsign"}), 2, "",
           usage_error("argument --callsign: expected one argument"), "no value");
    expect(run("beacon", {"a", "b", "--callsign", "-K1"}), 2, "",
           usage_error("argument --callsign: expected one argument"), "an option-like value");
    expect(run("beacon", {"a", "b", "--callsign", "K1", "--segment", "x"}), 2, "",
           usage_error("argument --segment: invalid int value: 'x'"), "type=int");
    expect(run("beacon", {"a", "b", "--callsign", "K1", "--ook=yes"}), 2, "",
           usage_error("argument --ook: ignored explicit argument 'yes'"), "store_true with a value");

    // A parser with two options sharing a prefix, as qrss_encode.py's.
    cli::ArgParser ap("p", "");
    ap.positional("image");
    ap.option("--model");
    ap.option("--mode", cli::ArgSpec().one_of({"A", "B", "C"}).otherwise("A"));
    ap.option("--level", cli::ArgSpec().real().otherwise("1.5"));
    ap.flag("--loud");
    auto parse = [&](const std::vector<std::string>& args, int& status, std::string& err) {
        std::ostringstream o, e;
        auto a = ap.parse(args, o, e, status);
        err = e.str();
        return a;
    };
    int status = 0;
    std::string err;
    check::is_true(!parse({"x", "--mod", "m"}, status, err), "ambiguous");
    check::equal(status, 2, "ambiguous: status");
    check::equal(err.substr(err.find("p: error: ")),
                 std::string("p: error: ambiguous option: --mod could match --model, --mode\n"), "ambiguous: text");
    parse({"x", "--mode", "D"}, status, err);
    check::equal(err.substr(err.find("p: error: ")),
                 std::string("p: error: argument --mode: invalid choice: 'D' (choose from 'A', 'B', 'C')\n"),
                 "choices");
    parse({"x", "--level", "1.2.3"}, status, err);
    check::equal(err.substr(err.find("p: error: ")),
                 std::string("p: error: argument --level: invalid float value: '1.2.3'\n"), "type=float");
    parse({"x", "--mo=B"}, status, err);
    check::equal(err.substr(err.find("p: error: ")),
                 std::string("p: error: ambiguous option: --mo=B could match --model, --mode\n"),
                 "ambiguous with a value");
    auto a = parse({"--mode=B", "--lo", "--", "-x"}, status, err);
    check::is_true(a.has_value(), "abbreviation, --opt=V and --");
    if (a) {
        check::equal(*a->text("image"), std::string("-x"), "after --");
        check::equal(*a->text("mode"), std::string("B"), "--mode=B");
        check::is_true(a->on("loud"), "--lo");
        check::is_true(!a->text("model").has_value(), "default None");
        check::equal(*a->real("level"), 1.5, "default as its type");
    }
    a = parse({"x", "--level", "-2.5", "--model", " -a b"}, status, err);
    check::is_true(a.has_value() && *a->real("level") == -2.5 && *a->text("model") == " -a b",
                   "a negative number and a value with a space are values");
    // argparse's negative-number pattern is ^-\d+$|^-\d*\.\d+$: "-2e1" is an option.
    parse({"x", "--level", "-2e1"}, status, err);
    check::equal(err.substr(err.find("p: error: ")),
                 std::string("p: error: argument --level: expected one argument\n"), "-2e1 is not a number");

    std::ostringstream o, e;
    check::is_true(!ap.parse({"x", "-h"}, o, e, status), "help");
    check::equal(status, 0, "help: status");
    check::is_true(o.str().rfind("usage: p [-h] [--model MODEL] [--mode {A,B,C}] [--level LEVEL] [--loud] image\n",
                                 0) == 0,
                   "help starts with the usage line");
    check::equal(e.str(), std::string(), "help: nothing on stderr");
}

// --- beacon ----------------------------------------------------------------------------

void test_beacon(const TempDir& t) {
    const pc::StoredPicture sp = closed_form_picture();
    check::equal(sp.picture_id, std::int64_t(0x73EA4EF6), "the picture");
    const std::string in = t / "pic.qrsp";
    pc::save_qrsp(in, sp);

    struct Case {
        std::vector<std::string> extra;
        const char* line;
        const char* digest;
    };
    // test_beaconfile.cpp's digests: hashlib.sha256(bf.to_bytes()) from the reference.
    const Case cases[] = {
        {{"--grid", "FN42"}, "picture 73ea4ef6 segment 0, K1ABC/P FN42, FSK ID, 50970 bytes",
         "2abe695a8676e64f09fb30c5b104ec336c168faa746a26e336bbe6dc3caabc77"},
        {{}, "picture 73ea4ef6 segment 0, K1ABC/P -, FSK ID, 50970 bytes",
         "9ccbcab0280eccc2fb2bc6b2380cd8af35eb1eb035787e1fb8e07900abfc4e42"},
        {{"--ook", "--gr=FN42"}, "picture 73ea4ef6 segment 0, K1ABC/P FN42, OOK ID, 50970 bytes",
         "df54ee3b013133f0d6ddc3f049ff540c15ce591a17d65fc011b7e8c4d0b60336"},
        {{"--segment", "1", "--grid", "FN42"}, "picture 73ea4ef6 segment 1, K1ABC/P FN42, FSK ID, 50970 bytes",
         "b8d3fd8666f6986392122373010c3217cf7b95f4b3f0116544dc9b6749a184f4"},
        {{"--seg", " +1_0 ", "--grid", "FN42"}, nullptr, nullptr},   // int(" +1_0 ") = 10: refused below
        {{"--segment=1", "--ook"}, "picture 73ea4ef6 segment 1, K1ABC/P -, OOK ID, 50970 bytes",
         "181758b4d59795f1d40eca7b36e0facfcf88be73fc90b7d0f49139d1098c5490"},
    };
    int k = 0;
    for (const Case& c : cases) {
        const std::string out = t / ("seg" + std::to_string(k++) + ".bin");
        std::vector<std::string> args{in, out, "--callsign", "K1ABC/P"};
        args.insert(args.end(), c.extra.begin(), c.extra.end());
        const Result r = run("beacon", args);
        if (!c.line) {
            expect(r, 1, "", "segment must be 0..1, not 10\n", "segment 10");
            check::is_true(!exists(out), "no file for a refusal");
            continue;
        }
        expect(r, 0, "wrote " + out + ": " + c.line + "\n", "", c.line);
        qrss_check::digest_is(slurp(out), c.digest, std::string("beacon file: ") + c.line);
    }

    const std::string out = t / "x.bin";
    expect(run("beacon", {in, out, "--callsign", "K1ABC", "--segment", "-1"}), 1, "",
           "segment must be 0..1, not -1\n", "segment -1");
    expect(run("beacon", {in, out, "--callsign", "K1ABC", "--segment", "99999999999999999999"}), 1, "",
           "segment must be 0..1, not 99999999999999999999\n", "a segment past int64");
    check::is_true(!exists(out), "no file for a refusal");
    const Result bad_call = run("beacon", {in, out, "--callsign", "K1ABC!"});
    check::equal(bad_call.status, 1, "bad callsign: a SystemExit");
    check::is_true(bad_call.err.find("Traceback") == std::string::npos && !bad_call.err.empty(),
                   "bad callsign: its text");

    const std::string missing = t / "missing.qrsp";
    expect(run("beacon", {missing, out, "--callsign", "K1ABC"}), 1, "",
           "FileNotFoundError: [Errno 2] No such file or directory: '" + missing + "'\n", "missing input");
    const std::string empty = t / "empty.qrsp";
    std::ofstream(empty).close();
    expect(run("beacon", {empty, out, "--callsign", "K1ABC"}), 1, "", "EOFError: No data left in file\n",
           "empty input");
    expect(run("beacon", {in, t / "nodir/x.bin", "--callsign", "K1ABC"}), 1, "",
           "FileNotFoundError: [Errno 2] No such file or directory: '" + (t / "nodir/x.bin") + "'\n",
           "unwritable output");
    check::is_true(!exists(out), "still no file");
}

// --- encode, as far as no model is needed -------------------------------------------------

void test_encode(const TempDir& t) {
    if (!cli::Subcommand::find("encode")) {
        std::printf("note: built without the codec; encode is not registered\n");
        return;
    }
    const std::string png = t / "p.png";
    std::ofstream(png) << "not really a picture, but a file";
    const std::string out = t / "o.qrsp";
    const std::string usage =
        "usage: sstvae-qrss encode [-h] [--model MODEL] [--precision {fp32,fp16,int8}] [--mode {A,B,C}] "
        "[--codec-id HEX] image output\n";
    expect(run("encode", {png}), 2, "",
           usage + "sstvae-qrss encode: error: the following arguments are required: output\n", "no output");
    expect(run("encode", {png, out, "--mod", "B"}), 2, "",
           usage + "sstvae-qrss encode: error: ambiguous option: --mod could match --model, --mode\n", "ambiguous");
    expect(run("encode", {png, out, "--precision", "fp8"}), 2, "",
           usage + "sstvae-qrss encode: error: argument --precision: invalid choice: 'fp8' (choose from 'fp32', "
                   "'fp16', 'int8')\n",
           "precision");
    expect(run("encode", {t / "nope.png", out}), 1, "", (t / "nope.png") + ": no such file\n", "no image");
    expect(run("encode", {t.path.string(), out}), 1, "", t.path.string() + ": no such file\n", "a directory");
    expect(run("encode", {png, out, "--codec-id", "xyz"}), 1, "",
           "ValueError: invalid literal for int() with base 16: 'xyz'\n", "codec ID not hex");
    expect(run("encode", {png, out, "--codec-id", "10000"}), 1, "", "--codec-id must be 16 bits, not 10000\n",
           "codec ID too big");
    expect(run("encode", {png, out, "--codec-id", "-1"}), 1, "", "--codec-id must be 16 bits, not -1\n",
           "codec ID negative");
    expect(run("encode", {png, out, "--model", "nowhere"}), 1, "",
           "--model nowhere: expected a .pt checkpoint, a .onnx artifact, or a directory containing the exported "
           ".onnx files\n",
           "--model neither");
    expect(run("encode", {png, out, "--model", t.path.string()}), 1, "",
           "no *-encoder-fp16.onnx in " + t.path.string() + "\nthat directory holds no .onnx files at all\n",
           "--model an empty directory");
    expect(run("encode", {png, out, "--model", "x.pt", "--codec-id", "d1d8"}), 1, "",
           "ValueError: precision applies to the ONNX backend only\n", "a .pt with a precision");
    const Result unreadable = run("encode", {png, out, "--model", t / "m", "--codec-id", "d1d8"});
    check::equal(unreadable.status, 1, "an unreadable image: status");
    check::is_true(unreadable.err.rfind(png + ": ", 0) == 0, "an unreadable image: '<image>: <reason>'");
    check::is_true(!exists(out), "no output for any refusal");
}

}  // namespace

int main() {
    try {
        const TempDir t;
        test_registry();
        test_argparse();
        test_beacon(t);
        test_encode(t);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return check::report("qrss_cli");
}
