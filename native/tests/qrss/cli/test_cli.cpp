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
// runs both programs side by side, the encode included. transmit is held
// to the sha256 of what qrss_transmit.py wrote for the same picture at
// slot 2025-10-01T00:00Z (q=1954752): a .qrsp and an OOK segment-1 beacon
// file, CE and L, int16 and float, plus its refusals, each of which must
// leave no output file behind.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include "argparse.hpp"
#include "check.hpp"
#include "common.hpp"
#include "qrss/beaconfile.hpp"
#include "qrss/header.hpp"
#include "qrss/npz.hpp"
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
    // The "p: error: ..." line, or a note that the parse succeeded.
    auto error_line = [](const std::string& e) {
        const std::size_t at = e.find("p: error: ");
        return at == std::string::npos ? "(no error)" + e : e.substr(at);
    };
    int status = 0;
    std::string err;
    check::is_true(!parse({"x", "--mod", "m"}, status, err), "ambiguous");
    check::equal(status, 2, "ambiguous: status");
    check::equal(error_line(err),
                 std::string("p: error: ambiguous option: --mod could match --model, --mode\n"), "ambiguous: text");
    parse({"x", "--mode", "D"}, status, err);
    check::equal(error_line(err),
                 std::string("p: error: argument --mode: invalid choice: 'D' (choose from 'A', 'B', 'C')\n"),
                 "choices");
    parse({"x", "--level", "1.2.3"}, status, err);
    check::equal(error_line(err),
                 std::string("p: error: argument --level: invalid float value: '1.2.3'\n"), "type=float");
    parse({"x", "--mo=B"}, status, err);
    check::equal(error_line(err),
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
    check::equal(error_line(err),
                 std::string("p: error: argument --level: expected one argument\n"), "-2e1 is not a number");

    // '--' is dropped only where a positional takes it (argparse's -*A-*);
    // once the positionals are filled it is an unrecognized argument.
    // Outcomes from CPython 3.13's argparse on this parser.
    parse({"x", "--loud", "--"}, status, err);
    check::equal(error_line(err), std::string("p: error: unrecognized arguments: --\n"),
                 "a trailing -- after the positionals");
    parse({"x", "--loud", "--", "y"}, status, err);
    check::equal(error_line(err), std::string("p: error: unrecognized arguments: -- y\n"),
                 "-- and a value after the positionals");
    parse({"x", "--", "y"}, status, err);
    check::equal(error_line(err), std::string("p: error: unrecognized arguments: y\n"),
                 "-- straight after a positional's value");
    a = parse({"--loud", "x", "--"}, status, err);
    check::is_true(a.has_value() && *a->text("image") == "x", "-- after the positional's value is its");
    a = parse({"--loud", "--", "--"}, status, err);
    check::is_true(a.has_value() && *a->text("image") == "--", "a second -- is a value");
    parse({"x", "-="}, status, err);
    check::equal(error_line(err),
                 std::string("p: error: ambiguous option: -= could match -h, --help, --model, --mode, --level, "
                             "--loud\n"),
                 "-= prefixes every option");
    a = parse({"x", "--level", "-2\n"}, status, err);
    check::is_true(a.has_value() && *a->real("level") == -2.0, "re's $ matches before a final newline");

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
    expect(run("beacon", {in, out, "--callsign", "K1ABC", "--"}), 2, "", usage_error("unrecognized arguments: --"),
           "a trailing -- once the positionals are filled");
    check::is_true(!exists(out), "no file for a usage error");
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
    // A .qrsp whose scalars int() cannot read: the class is numpy's or
    // Python's, as the traceback's last line names it (numpy 2.5.3).
    auto with_version = [&](const std::string& name, sstvae::qrss::npz::Array v) {
        const sstvae::qrss::npz::Npz good = pc::qrsp_npz(sp);
        sstvae::qrss::npz::Npz z;
        for (const auto& [member, a] : good.members()) z.add(member, member == "version" ? v : a);
        const std::string p = t / name;
        sstvae::qrss::npz::save_npz(p, z);
        return p;
    };
    namespace npz = sstvae::qrss::npz;
    expect(run("beacon", {with_version("v1d.qrsp", npz::Array::of(std::vector<std::int64_t>{1, 1})), out,
                          "--callsign", "K1ABC"}),
           1, "", "TypeError: only 0-dimensional arrays can be converted to Python scalars\n", "a 1-d version");
    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    expect(run("beacon", {with_version("vinf.qrsp", npz::Array::scalar<double>(inf)), out, "--callsign", "K1ABC"}),
           1, "", "OverflowError: cannot convert float infinity to integer\n", "an infinite version");
    expect(run("beacon", {with_version("vnan.qrsp", npz::Array::scalar<double>(nan)), out, "--callsign", "K1ABC"}),
           1, "", "ValueError: cannot convert float NaN to integer\n", "a NaN version");
    expect(run("beacon", {with_version("v2.qrsp", npz::Array::scalar<std::int64_t>(2)), out, "--callsign",
                          "K1ABC"}),
           1, "", "ValueError: " + (t / "v2.qrsp") + ": .qrsp version 2, expected 1\n", "version 2");
    check::is_true(!exists(out), "no file for a malformed picture");
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
    // resolve_onnx's glob takes a directory named like the encoder, and
    // codec_id_from_onnx then finds neither part inside it: the reference's
    // uncaught FileNotFoundError (this used to escape as std::terminate).
    fs::create_directories(t.path / "dd" / "v9-encoder-fp16.onnx");
    expect(run("encode", {png, out, "--model", t / "dd"}), 1, "",
           "FileNotFoundError: no *-encoder-fp16.onnx or *-decoder-fp16.onnx in " + (t / "dd/v9-encoder-fp16.onnx") +
               "\n",
           "an encoder that is a directory");
    const Result unreadable = run("encode", {png, out, "--model", t / "m", "--codec-id", "d1d8"});
    check::equal(unreadable.status, 1, "an unreadable image: status");
    check::is_true(unreadable.err.rfind(png + ": ", 0) == 0, "an unreadable image: '<image>: <reason>'");
    check::is_true(!exists(out), "no output for any refusal");
}

// --- transmit ------------------------------------------------------------------------------

void test_transmit(const TempDir& t) {
    namespace bfm = sstvae::qrss::beaconfile;
    const pc::StoredPicture sp = closed_form_picture();
    const std::string pic = t / "tx.qrsp";
    pc::save_qrsp(pic, sp);
    sstvae::qrss::header::HeaderFields h;
    h.callsign = "K1ABC/P";
    h.grid = std::string("FN42");
    h.picture_id = sp.picture_id;
    h.mode = sp.mode;
    h.segment = 1;
    h.codec_id = sp.codec_id;
    const std::string seg1 = t / "tx-seg1.bin";
    bfm::write(seg1, bfm::from_picture(sp, 1, h, true));

    const std::string slot = "2025-10-01T00:00Z";
    struct Case {
        std::string input;
        std::vector<std::string> extra;
        const char* line;
        const char* digest;
    };
    // hashlib.sha256 of the WAV qrss_transmit.py wrote, and its stdout after "wrote <out>: ".
    const Case cases[] = {
        {pic,
         {"--callsign", "K1ABC/P", "--frame", "tiny", "--float"},
         "slot q=1954752, waveform CE, frame tiny (53.1 s), K1ABC/P (FSK callsign windows), carrier 1500 Hz, "
         "68.3 s of audio starting 12 s before t0",
         "46202a2e720b3d49bd29c1f2027b9608942fafeef6e2a3ef1c4daf6fe5c60089"},
        {pic,
         {"--callsign", "K1ABC/P", "--frame", "tiny", "--waveform", "L", "--lead-in", "2.5"},
         "slot q=1954752, waveform L, frame l-tiny (53.2 s), K1ABC/P (FSK callsign windows), pilot 1500 Hz, "
         "2.5 s lead-in, 68.6 s of audio starting 12 s before t0",
         "a5faee5c8a1a9aa64adae26c8899098995b2d73ed5d76432c257e6f228d023d1"},
        {pic,
         {"--callsign", "K1ABC/P", "--frame", "short", "--grid", "FN42", "--segment", "1", "--ook", "--freq",
          "1234.5"},
         "slot q=1954752, waveform CE, frame short (255.8 s), K1ABC/P (OOK callsign windows), carrier 1234.5 Hz, "
         "270.8 s of audio starting 12 s before t0",
         "74f01cf22ef2f63c6ceb10405fa3215b7f9d67005a8aed3490e1622415cd53ca"},
        {seg1,
         {"--frame", "tiny", "--float"},
         "slot q=1954752, waveform CE, frame tiny (53.1 s), K1ABC/P (OOK callsign windows), carrier 1500 Hz, "
         "68.3 s of audio starting 12 s before t0",
         "1dd8331f27eac768d252f286038ec7c3d035b45534deefaa6eb23dfb7a99c1f1"},
    };
    int k = 0;
    for (const Case& c : cases) {
        const std::string out = t / ("tx" + std::to_string(k++) + ".wav");
        std::vector<std::string> args{c.input, out, "--slot", slot};
        args.insert(args.end(), c.extra.begin(), c.extra.end());
        expect(run("transmit", args), 0, "wrote " + out + ": " + c.line + "\n", "", c.line);
        qrss_check::digest_is(slurp(out), c.digest, std::string("transmit audio: ") + c.line);
    }

    // The refusals, as qrss_transmit.py words them; none may leave a file.
    const std::string out = t / "refused.wav";
    struct Refusal {
        std::string input;
        std::vector<std::string> extra;
        std::string err;
    };
    const Refusal refusals[] = {
        {pic,
         {"--slot", "2025-10-01T00:05Z", "--callsign", "K1ABC"},
         "--slot '2025-10-01T00:05Z': 2025-10-01T00:05:00+00:00 is not on a quarter hour\n"},
        {pic,
         {"--slot", "nope", "--callsign", "K1ABC"},
         "--slot 'nope': not an ISO 8601 time; expected e.g. 2026-10-09T06:00Z\n"},
        {pic, {"--slot", slot}, "a .qrsp needs --callsign: every pass sends it in the header and in the callsign windows\n"},
        {pic,
         {"--slot", slot, "--callsign", "K1ABC", "--waveform", "L", "--ook"},
         "--ook is waveform CE's: waveform L keys its callsign windows by FSK of its pilot\n"},
        {pic,
         {"--slot", "1969-12-31T23:45Z", "--callsign", "K1ABC"},
         "--slot '1969-12-31T23:45Z': slot 1969-12-31T23:45Z is before 1970-01-01T00:00Z\n"},
        {seg1, {"--slot", slot, "--callsign", "K1ABC"}, "the beacon file is for K1ABC/P, not K1ABC\n"},
        {pic, {"--slot", slot, "--callsign", "K1ABC", "--segment", "2"}, "segment must be 0..1, not 2\n"},
        {pic,
         {"--slot", slot, "--callsign", "K1ABC", "--freq", "100"},
         "carrier 100.0 Hz is outside the 300-2700 Hz audio band that receivers search\n"},
        {pic, {"--slot", slot, "--callsign", "K1ABC", "--lead-in", "11"}, "lead-in must be 0 to 10 s, not 11.0\n"},
    };
    for (const Refusal& r : refusals) {
        std::vector<std::string> args{r.input, out};
        args.insert(args.end(), r.extra.begin(), r.extra.end());
        expect(run("transmit", args), 1, "", r.err, "transmit refusal: " + r.err);
        check::is_true(!exists(out), "no file for a refusal: " + r.err);
    }
}

}  // namespace

int main() {
    try {
        const TempDir t;
        test_registry();
        test_argparse();
        test_beacon(t);
        test_transmit(t);
        test_encode(t);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return check::report("qrss_cli");
}
