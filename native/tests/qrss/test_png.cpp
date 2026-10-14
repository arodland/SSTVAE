// native/core/qrss/png.hpp: a picture written by its UTF-8 name through
// util/fileio.hpp, its bytes exactly images::save_png's.
//
// What runs here is the byte identity (several pictures, against the file
// save_png writes), a name outside ASCII (on Windows the reason png.cpp
// exists -- a narrow fopen cannot open it -- compiled by the mingw
// cross-check; here it shows the name reaches the file system as its UTF-8
// bytes), the temporary sibling's name and its removal on every failure,
// and the errors.

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include "check.hpp"
#include "images/images.hpp"
#include "qrss/png.hpp"
#include "qrss/rawfile.hpp"
#include "util/fileio.hpp"

using namespace sstvae;
namespace fs = std::filesystem;
namespace png = sstvae::qrss::png;

namespace {

std::vector<std::uint8_t> read_bytes(const std::string& path) {
    util::File f = util::open_file(path, "rb");
    std::vector<std::uint8_t> out;
    if (!f) return out;
    std::uint8_t buf[1 << 14];
    for (std::size_t n; (n = std::fread(buf, 1, sizeof buf, f.get())) > 0;) out.insert(out.end(), buf, buf + n);
    return out;
}

images::Picture noise(int w, int h, unsigned seed) {
    images::Picture p(w, h);
    std::mt19937 rng(seed);
    for (auto& v : p.rgb) v = static_cast<std::uint8_t>(rng() & 0xFF);
    return p;
}

images::Picture gradient(int w, int h) {
    images::Picture p(w, h);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            std::uint8_t* px = p.rgb.data() + (static_cast<std::size_t>(y) * w + x) * 3;
            px[0] = static_cast<std::uint8_t>(x * 255 / (w > 1 ? w - 1 : 1));
            px[1] = static_cast<std::uint8_t>(y * 255 / (h > 1 ? h - 1 : 1));
            px[2] = static_cast<std::uint8_t>((x ^ y) & 0xFF);
        }
    return p;
}

// The names in `dir` that are not `keep`.
std::vector<std::string> others(const fs::path& dir, const std::string& keep) {
    std::vector<std::string> out;
    for (const auto& e : fs::directory_iterator(dir)) {
        const std::string n = util::path_text(e.path().filename());
        if (n != keep) out.push_back(n);
    }
    return out;
}

void test_bytes_are_save_pngs(const fs::path& dir) {
    struct Case {
        const char* what;
        images::Picture pic;
    };
    const std::vector<Case> cases = {
        {"640x480 noise", noise(640, 480, 1)},
        {"640x480 gradient", gradient(640, 480)},
        {"320x240 flat", images::Picture(320, 240)},
        {"1x1", noise(1, 1, 2)},
        {"7x3 (an odd stride)", noise(7, 3, 3)},
        {"3x500 (tall)", gradient(3, 500)},
    };
    for (const Case& c : cases) {
        const std::string ref = util::path_text(dir / "ref.png");
        images::save_png(c.pic, ref);
        const std::vector<std::uint8_t> want = read_bytes(ref);
        check::is_true(want.size() > 8, std::string(c.what) + ": save_png wrote a file");
        check::is_true(png::png_bytes(c.pic) == want, std::string(c.what) + ": png_bytes is save_png's file");
        const std::string mine = util::path_text(dir / "mine.png");
        png::write_png(c.pic, mine);
        check::is_true(read_bytes(mine) == want, std::string(c.what) + ": write_png writes save_png's bytes");
        fs::remove(dir / "ref.png");
        fs::remove(dir / "mine.png");
        check::equal(others(dir, "").size(), std::size_t(0), std::string(c.what) + ": no temporary left");
    }
}

void test_a_name_outside_ascii(const fs::path& dir) {
    // "Ünïcødé ✓/tile – α.png" as UTF-8: the file is found under exactly
    // those bytes (on POSIX a name is bytes; on Windows fileio makes it
    // UTF-16, where a narrow fopen would read it through the ANSI code page)
    const std::string sub = util::path_text(dir) + "/\xC3\x9Cn\xC3\xAF" "c\xC3\xB8" "d\xC3\xA9 \xE2\x9C\x93";
    std::error_code ec;
    fs::create_directories(*util::fs_path(sub), ec);
    check::is_true(!ec, "made the directory");
    const std::string name = "tile \xE2\x80\x93 \xCE\xB1.png";
    const images::Picture pic = noise(64, 48, 9);
    png::write_png(pic, sub + "/" + name);
    check::is_true(util::is_file(sub + "/" + name), "the picture is there under its UTF-8 name");
    check::is_true(read_bytes(sub + "/" + name) == png::png_bytes(pic), "its bytes");
    const auto rest = others(*util::fs_path(sub), name);
    check::equal(rest.size(), std::size_t(0), "nothing else in the directory");
    // a second write replaces it
    const images::Picture pic2 = gradient(64, 48);
    png::write_png(pic2, sub + "/" + name);
    check::is_true(read_bytes(sub + "/" + name) == png::png_bytes(pic2), "replaced");
}

void test_failures(const fs::path& dir) {
    const images::Picture pic = noise(16, 16, 4);
    // no such directory: the temporary cannot be opened (ENOENT, naming it)
    const std::string missing = util::path_text(dir / "nope" / "x.png");
    try {
        png::write_png(pic, missing);
        check::fail("a missing directory", "no error");
    } catch (const qrss::rawfile::FileError& e) {
        check::is_true(e.code().category() == std::generic_category(), "an errno");
        check::equal(e.code().value(), ENOENT, "ENOENT");
        check::is_true(e.path().find("/nope/.x.png.") != std::string::npos && e.path().size() > 4 &&
                           e.path().substr(e.path().size() - 4) == ".tmp",
                       "names the temporary: " + e.path());
    }
    // the target is a directory: the move fails, the temporary is removed
    fs::create_directories(dir / "adir" / "x.png");
    try {
        png::write_png(pic, util::path_text(dir / "adir" / "x.png"));
        check::fail("a directory in the way", "no error");
    } catch (const qrss::rawfile::FileError& e) {
        check::is_true(e.code().category() == std::generic_category() && e.code().value() != 0,
                       "an errno: " + std::to_string(e.code().value()));
    }
    check::equal(others(dir / "adir", "x.png").size(), std::size_t(0), "the temporary removed after a failed move");
    // a buffer that does not hold the picture is refused before anything is written
    images::Picture bad(4, 4);
    bad.rgb.resize(10);
    try {
        png::write_png(bad, util::path_text(dir / "bad.png"));
        check::fail("a short buffer", "no error");
    } catch (const std::runtime_error& e) {
        check::is_true(std::string(e.what()).find("cannot encode a 4x4 picture") == 0, e.what());
    }
    check::is_true(!util::exists(util::path_text(dir / "bad.png")), "nothing written");
    // the temporary's name: store.py's _tmp_name, ".<name>.<pid>.<thread id>.tmp"
    const std::string want = "." + std::string("x.png") + "." + std::to_string(qrss::rawfile::process_id()) + "." +
                             std::to_string(qrss::rawfile::thread_ident()) + ".tmp";
    try {
        png::write_png(pic, missing);
    } catch (const qrss::rawfile::FileError& e) {
        check::is_true(e.path().size() >= want.size() &&
                           e.path().substr(e.path().size() - want.size()) == want,
                       "the temporary is " + want);
    }
}

}  // namespace

int main() {
    const fs::path dir = fs::temp_directory_path() / ("sstvae_png_" + std::to_string(std::random_device{}()));
    fs::create_directories(dir);
    try {
        const fs::path a = dir / "a", b = dir / "b", c = dir / "c";
        fs::create_directories(a);
        fs::create_directories(b);
        fs::create_directories(c);
        test_bytes_are_save_pngs(a);
        test_a_name_outside_ascii(b);
        test_failures(c);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        fs::remove_all(dir);
        return 1;
    }
    fs::remove_all(dir);
    return check::report("qrss_core_png");
}
