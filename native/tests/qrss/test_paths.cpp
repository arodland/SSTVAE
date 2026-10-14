// File names in the native code (native/core/util/fileio.hpp) and how the
// qrss_*.py scripts spell them (native/core/qrss/pypath.hpp).
//
// - The UTF-8 <-> UTF-16 conversion Windows opens files through, against
//   CPython's "utf-8"/"surrogatepass" codec (PEP 529), by the digests
//   gen_paths_pins.py wrote: every code point both ways, and every byte
//   string of one and two bytes, every three-byte string from E0..EF and a
//   four-byte family from F0..F4 decoded (or refused) as Python does. Tested
//   here, on every platform, because the one machine that runs it for real
//   is the Windows CI job.
// - open_file / exists / is_file / is_dir / list_dir / remove_file /
//   replace_file on names that are not ASCII and (POSIX) not UTF-8 at all,
//   in a scratch directory: a name goes to the system as its bytes, and
//   comes back the same.
// - pypath against pathlib.PurePosixPath's str, name and suffix,
//   stderr_text against what the scripts' sys.stderr writes for a message
//   naming a file whose name is not UTF-8, and fs_decode against the str
//   os.fsdecode holds for it (what a refusal's repr names).

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "check.hpp"
#include "paths_pins.hpp"
#include "qrss/pypath.hpp"
#include "qrss/pytext.hpp"
#include "util/fileio.hpp"
#include "util/sha256.hpp"

namespace check = sstvae::check;
namespace util = sstvae::util;
namespace pypath = sstvae::qrss::pypath;
namespace pytext = sstvae::qrss::pytext;
namespace fs = std::filesystem;

namespace {

std::string le_bytes(const std::u16string& u) {
    std::string out;
    for (const char16_t c : u) {
        out += static_cast<char>(c & 0xFF);
        out += static_cast<char>(c >> 8);
    }
    return out;
}

std::string hex_of(util::Sha256& h) {
    const util::Sha256Digest d = h.digest();
    return util::to_hex(d);
}

// --- the conversion --------------------------------------------------------------------

void test_code_points() {
    util::Sha256 enc, u16;
    std::size_t round_trips = 0;
    for (std::uint32_t cp = 0; cp < 0x110000; ++cp) {
        std::u16string u;
        if (cp < 0x10000) {
            u += static_cast<char16_t>(cp);
        } else {
            u += static_cast<char16_t>(0xD800 + ((cp - 0x10000) >> 10));
            u += static_cast<char16_t>(0xDC00 + ((cp - 0x10000) & 0x3FF));
        }
        const std::string utf8 = util::utf8_of(u);
        enc.update(utf8);
        const std::optional<std::u16string> back = util::utf16_of(utf8);
        if (back && *back == u) ++round_trips;
        u16.update(le_bytes(back ? *back : std::u16string()));
    }
    check::equal(hex_of(enc), std::string(paths_pins::CODE_POINTS_UTF8), "every code point to UTF-8");
    check::equal(hex_of(u16), std::string(paths_pins::CODE_POINTS_UTF16), "and back to UTF-16");
    check::equal(round_trips, std::size_t{0x110000}, "every code point round-trips");
}

struct Family {
    util::Sha256 h;
    std::size_t n = 0, ok = 0;
    void add(const std::string& bytes) {
        ++n;
        const std::optional<std::u16string> u = util::utf16_of(bytes);
        if (!u) {
            h.update(std::string_view("\0", 1));
            return;
        }
        const std::string le = le_bytes(*u);
        std::string rec = "\x01";
        rec += static_cast<char>(le.size());
        h.update(rec + le);
        ++ok;
        // what decodes, encodes back to itself (one byte string per name)
        if (util::utf8_of(*u) != bytes) check::fail("utf8_of(utf16_of(b)) == b", bytes);
    }
};

void family_is(Family& f, const char* digest, std::size_t count, std::size_t accepted, const std::string& what) {
    check::equal(f.n, count, what + " strings");
    check::equal(f.ok, accepted, what + " accepted");
    check::equal(hex_of(f.h), std::string(digest), what + " digest");
}

void test_byte_strings() {
    Family one_two;
    for (int a = 0; a < 256; ++a) one_two.add(std::string(1, static_cast<char>(a)));
    for (int a = 0; a < 256; ++a)
        for (int b = 0; b < 256; ++b) one_two.add({static_cast<char>(a), static_cast<char>(b)});
    family_is(one_two, paths_pins::ONE_TWO_DIGEST, paths_pins::ONE_TWO_COUNT, paths_pins::ONE_TWO_ACCEPTED,
              "one and two bytes");

    Family three;
    for (int a = 0xE0; a < 0xF0; ++a)
        for (int b = 0; b < 256; ++b)
            for (int c = 0; c < 256; ++c) three.add({static_cast<char>(a), static_cast<char>(b), static_cast<char>(c)});
    family_is(three, paths_pins::THREE_DIGEST, paths_pins::THREE_COUNT, paths_pins::THREE_ACCEPTED, "three bytes");

    Family four;
    for (int a = 0xF0; a < 0xF5; ++a)
        for (int b = 0; b < 256; ++b)
            for (const unsigned char c : paths_pins::EDGES)
                for (const unsigned char d : paths_pins::EDGES)
                    four.add({static_cast<char>(a), static_cast<char>(b), static_cast<char>(c), static_cast<char>(d)});
    family_is(four, paths_pins::FOUR_DIGEST, paths_pins::FOUR_COUNT, paths_pins::FOUR_ACCEPTED, "four bytes");

    // A few by name, so a failure above has a readable neighbour.
    check::is_true(util::utf16_of("\xED\xA0\x80") == std::u16string(1, char16_t{0xD800}), "surrogatepass D800");
    check::is_true(!util::utf16_of("\xC0\x80"), "overlong NUL refused");
    check::is_true(!util::utf16_of("\xF4\x90\x80\x80"), "past U+10FFFF refused");
    check::is_true(!util::utf16_of("caf\xE9"), "Latin-1 refused");
    check::equal(util::utf8_of(std::u16string(u"\xD83D\xDE00")), std::string("\xF0\x9F\x98\x80"), "a pair is one code point");
}

// --- the file system ---------------------------------------------------------------------

std::string scratch_dir() {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const fs::path p = fs::temp_directory_path() / ("sstvae-qrss-paths-" + std::to_string(stamp));
    fs::create_directories(p);
    return util::path_text(p);
}

bool write_text(const std::string& path, const std::string& text) {
    const util::File f = util::open_file(path, "wb");
    return f && std::fwrite(text.data(), 1, text.size(), f.get()) == text.size();
}

std::optional<std::string> read_text(const std::string& path) {
    const util::File f = util::open_file(path, "rb");
    if (!f) return std::nullopt;
    std::string out;
    char buf[256];
    for (std::size_t n; (n = std::fread(buf, 1, sizeof buf, f.get())) > 0;) out.append(buf, n);
    return out;
}

bool listed(const std::string& dir, const std::string& name) {
    std::error_code ec;
    for (const std::string& n : util::list_dir(dir, ec))
        if (n == name) return !ec;
    return false;
}

void test_names(const std::string& dir) {
    std::vector<std::string> names = {
        "plain.wav",
        "caf\xC3\xA9 \xE2\x80\x94 \xD0\xBF\xD1\x80\xD0\xB8\xD0\xB5\xD0\xBC.wav",   // é, an em dash, Cyrillic
        "\xE6\x97\xA5\xE6\x9C\xAC.npz",                                            // CJK
        "\xF0\x9F\x98\x80.qrsp",                                                   // outside the BMP
    };
#ifndef _WIN32
    // Not UTF-8 at all: a POSIX name is bytes (Python's surrogateescape).
    names.push_back("caf\xE9.wav");
    names.push_back("\xFF\xFE.npz");
#else
    // A lone surrogate, which an NTFS name may hold (Python: surrogatepass).
    names.push_back("x\xED\xA0\x80.wav");
#endif
    for (const std::string& name : names) {
        const std::string path = pypath::child(dir, name);
        check::is_true(!util::exists(path), name + ": not there before");
        check::is_true(write_text(path, "data:" + name), name + ": written");
        check::is_true(util::exists(path) && util::is_file(path) && !util::is_dir(path), name + ": a file");
        check::is_true(listed(dir, name), name + ": listed under its own bytes");
        const std::optional<std::string> back = read_text(path);
        check::is_true(back && *back == "data:" + name, name + ": read back");
#ifndef _WIN32
        // the system holds exactly these bytes
        const auto p = util::fs_path(path);
        check::is_true(p && fs::exists(fs::path(dir) / name), name + ": the system's name is its bytes");
#endif
        const std::string moved = pypath::child(dir, "moved-" + name);
        std::error_code ec;
        check::is_true(util::replace_file(path, moved, ec) && !ec, name + ": replaced");
        check::is_true(!util::exists(path) && util::is_file(moved), name + ": moved");
        check::is_true(util::remove_file(moved, ec) && !ec, name + ": removed");
        check::is_true(!util::exists(moved), name + ": gone");
    }

    // a directory with such a name
    const std::string sub = pypath::child(dir, "\xD0\xB4\xD0\xB8\xD1\x80");
    const auto subp = util::fs_path(sub);
    check::is_true(subp && fs::create_directory(*subp), "directory made");
    check::is_true(util::is_dir(sub) && !util::is_file(sub) && util::exists(sub), "a directory");
    check::is_true(write_text(pypath::child(sub, "a.onnx"), "x"), "a file inside it");
    std::error_code ec;
    const std::vector<std::string> inside = util::list_dir(sub, ec);
    check::is_true(!ec && inside == std::vector<std::string>{"a.onnx"}, "listed");
    check::is_true(!util::remove_file(sub, ec) && ec == std::errc::is_a_directory, "remove_file refuses a directory");
    check::is_true(util::remove_file(pypath::child(sub, "a.onnx"), ec), "its file removed");
    fs::remove(*subp);
}

void test_refusals(const std::string& dir) {
    const std::string nul("a\0b.wav", 7);
    errno = 0;
    check::is_true(!util::open_file(nul, "wb"), "a NUL in a name: not opened");
    check::equal(errno, EINVAL, "a NUL in a name: EINVAL");
    check::is_true(!util::exists(nul) && !util::is_file(nul) && !util::is_dir(nul), "a NUL in a name: not there");
    check::is_true(!util::exists(dir + "/a"), "nothing was made at the name before the NUL");
    check::is_true(!util::fs_path(nul), "a NUL in a name: no fs path");

    errno = 0;
    check::is_true(!util::open_file(pypath::child(dir, "missing.wav"), "rb"), "missing: not opened");
    check::equal(errno, ENOENT, "missing: ENOENT");
    std::error_code ec;
    check::is_true(!util::remove_file(pypath::child(dir, "missing.wav"), ec) && ec, "missing: not removed");
    check::is_true(util::list_dir(pypath::child(dir, "missing"), ec).empty() && ec, "missing dir: not listed");
#ifdef _WIN32
    errno = 0;
    check::is_true(!util::open_file(dir + "/caf\xE9.wav", "wb"), "not UTF-8 on Windows: not opened");
    check::equal(errno, EILSEQ, "not UTF-8 on Windows: EILSEQ");
#endif
}

// --- pathlib ---------------------------------------------------------------------------------

void test_pathlib() {
    for (const auto& p : paths_pins::PATHS) {
        const std::string t = p.text;
        check::equal(pypath::pathlib_str(t), std::string(p.str), "str(PurePosixPath(" + t + "))");
        check::equal(pypath::pathlib_name(t), std::string(p.name), "PurePosixPath(" + t + ").name");
        check::equal(pypath::pathlib_suffix(t), std::string(p.suffix), "PurePosixPath(" + t + ").suffix");
    }
    check::equal(pypath::child("a//b/", "c.onnx"), std::string("a/b/c.onnx"), "child");
    check::equal(pypath::child("", "c"), std::string("c"), "child of nothing");
    check::equal(pypath::with_name("m/x-encoder-fp16.onnx", "x-decoder-fp16.onnx"),
                 std::string("m/x-decoder-fp16.onnx"), "with_name");
    // Path(t).with_name("x")'s ValueError, for every path with no name
    for (const auto& [t, want] : {std::pair<const char*, const char*>{"/", "PosixPath('/') has an empty name"},
                                  {"", "PosixPath('.') has an empty name"},
                                  {"//", "PosixPath('//') has an empty name"}}) {
        std::string got = "(no exception)";
        try {
            pypath::with_name(t, "x");
        } catch (const std::invalid_argument& e) {
            got = e.what();
        }
        check::equal(got, std::string(want), std::string("with_name of ") + t + ": pathlib's ValueError text");
    }
}

// --- stderr ------------------------------------------------------------------------------

std::string stderr_once(const std::string& text) {
    std::string out;
    const std::size_t used = pypath::stderr_text(text, true, out);
    if (used != text.size()) check::fail("stderr_text(final) uses every byte", text);
    return out;
}

struct StderrFamily {
    util::Sha256 h, decoded;
    std::size_t streamed_ok = 0, n = 0, str_ok = 0;
    void add(const std::string& bytes) {
        ++n;
        const std::string out = stderr_once(bytes);
        std::string rec;
        rec += static_cast<char>(out.size() & 0xFF);
        rec += static_cast<char>(out.size() >> 8);
        h.update(rec + out);
        // fs_decode's code points, as UTF-32LE
        const std::u32string cps = pypath::fs_decode(bytes);
        // fs_str carries that str as pytext::decode reads it back
        str_ok += pytext::decode(pypath::fs_str(bytes)) == cps;
        std::string u32;
        for (const char32_t cp : cps)
            for (int k = 0; k < 4; ++k) u32 += static_cast<char>((static_cast<std::uint32_t>(cp) >> (8 * k)) & 0xFF);
        std::string drec;
        drec += static_cast<char>(u32.size() & 0xFF);
        drec += static_cast<char>(u32.size() >> 8);
        decoded.update(drec + u32);
        // the same bytes in two writes, split anywhere, give the same text
        bool same = true;
        for (std::size_t cut = 1; cut < bytes.size(); ++cut) {
            std::string got;
            std::string pending = bytes.substr(0, cut);
            pending.erase(0, pypath::stderr_text(pending, false, got));
            pending += bytes.substr(cut);
            pypath::stderr_text(pending, true, got);
            same = same && got == out;
        }
        streamed_ok += same;
    }
};

void stderr_family_is(StderrFamily& f, const char* posix, const char* windows, const char* decode_posix,
                      const char* decode_windows, const std::string& what) {
#ifdef _WIN32
    const char* digest = windows;
    const char* decode = decode_windows;
    (void)posix;
    (void)decode_posix;
#else
    const char* digest = posix;
    const char* decode = decode_posix;
    (void)windows;
    (void)decode_windows;
#endif
    check::equal(hex_of(f.h), std::string(digest), what + " stderr digest");
    check::equal(f.streamed_ok, f.n, what + " stderr in two writes");
    check::equal(hex_of(f.decoded), std::string(decode), what + " fs_decode digest");
    check::equal(f.str_ok, f.n, what + " fs_str read back as fs_decode");
}

void test_stderr() {
    StderrFamily one_two;
    for (int a = 0; a < 256; ++a) one_two.add(std::string(1, static_cast<char>(a)));
    for (int a = 0; a < 256; ++a)
        for (int b = 0; b < 256; ++b) one_two.add({static_cast<char>(a), static_cast<char>(b)});
    stderr_family_is(one_two, paths_pins::STDERR_ONE_TWO_POSIX, paths_pins::STDERR_ONE_TWO_WINDOWS,
                     paths_pins::DECODE_ONE_TWO_POSIX, paths_pins::DECODE_ONE_TWO_WINDOWS, "one and two bytes");

    StderrFamily three;
    for (int a = 0xE0; a < 0xF0; ++a)
        for (int b = 0; b < 256; ++b)
            for (int c = 0; c < 256; ++c) three.add({static_cast<char>(a), static_cast<char>(b), static_cast<char>(c)});
    stderr_family_is(three, paths_pins::STDERR_THREE_POSIX, paths_pins::STDERR_THREE_WINDOWS,
                     paths_pins::DECODE_THREE_POSIX, paths_pins::DECODE_THREE_WINDOWS, "three bytes");

    StderrFamily four;
    for (int a = 0xF0; a < 0xF5; ++a)
        for (int b = 0; b < 256; ++b)
            for (const unsigned char c : paths_pins::EDGES)
                for (const unsigned char d : paths_pins::EDGES)
                    four.add({static_cast<char>(a), static_cast<char>(b), static_cast<char>(c), static_cast<char>(d)});
    stderr_family_is(four, paths_pins::STDERR_FOUR_POSIX, paths_pins::STDERR_FOUR_WINDOWS,
                     paths_pins::DECODE_FOUR_POSIX, paths_pins::DECODE_FOUR_WINDOWS, "four bytes");

    // By name, so a failure above has a readable neighbour.
    check::equal(stderr_once("caf\xC3\xA9 \xE6\x97\xA5.wav"), std::string("caf\xC3\xA9 \xE6\x97\xA5.wav"), "UTF-8 as it is");
    check::equal(stderr_once("caf\xE9-\xFF\xFE"), std::string("caf\\udce9-\\udcff\\udcfe"), "bytes escaped");
#ifdef _WIN32
    check::equal(stderr_once("\xED\xA0\x80"), std::string("\\ud800"), "a lone surrogate (surrogatepass)");
#else
    check::equal(stderr_once("\xED\xA0\x80"), std::string("\\udced\\udca0\\udc80"), "a lone surrogate's bytes");
#endif
    // A refusal's repr(os.fsdecode(name)), as str(OSError) and argparse print it.
    check::equal(pytext::repr(pypath::fs_decode("caf\xE9.wav")), std::string("'caf\\udce9.wav'"), "repr: an escaped byte");
#ifdef _WIN32
    check::equal(pytext::repr(pypath::fs_decode("x\xED\xA0\x80")), std::string("'x\\ud800'"), "repr: a lone surrogate");
#else
    check::equal(pytext::repr(pypath::fs_decode("x\xED\xA0\x80")), std::string("'x\\udced\\udca0\\udc80'"),
                 "repr: a lone surrogate's bytes, escaped one by one");
#endif
#ifndef _WIN32
    check::equal(pypath::fs_str("x\xE9\xED\xA0\x80"), std::string("x\xED\xB3\xA9\xED\xB3\xAD\xED\xB2\xA0\xED\xB2\x80"),
                 "fs_str: each escape as its surrogate's three bytes");
#endif
    check::equal(pypath::fs_str("caf\xC3\xA9"), std::string("caf\xC3\xA9"), "fs_str: UTF-8 as it is");
    std::string out;
    check::equal(pypath::stderr_text("ok \xE6\x97", false, out), std::size_t{3}, "a cut-short sequence waits");
    check::equal(out, std::string("ok "), "and nothing of it is written");
}

}  // namespace

int main() {
    check::report_crashes_instead_of_prompting();
    std::string dir;
    try {
        test_code_points();
        test_byte_strings();
        test_pathlib();
        test_stderr();
        dir = scratch_dir();
        test_names(dir);
        test_refusals(dir);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    if (!dir.empty()) {
        std::error_code ec;
        if (const auto p = util::fs_path(dir)) fs::remove_all(*p, ec);
    }
    return check::report("qrss_paths");
}
