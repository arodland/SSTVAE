// native/core/util/fileio.hpp's errno_of / errno_code_of (an error code as
// the errno an OSError carries) and replace_file (os.replace). The Windows
// half -- a Win32 code through the category's portable condition, and
// replace_file's retry on ERROR_ACCESS_DENIED / ERROR_SHARING_VIOLATION --
// is compiled by the mingw cross-check; what runs here is the logic every
// platform shares, with system_category's Win32 meaning stood in for by a
// category of the test's own.

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <system_error>
#include <utility>

#include "check.hpp"
#include "util/fileio.hpp"

using namespace sstvae;

namespace {

// A category whose codes map to errno conditions as MSVC's system_category
// maps Win32 codes (5 -> EACCES, 2 -> ENOENT, 1234 -> none).
class FakeWin32 : public std::error_category {
public:
    const char* name() const noexcept override { return "fake-win32"; }
    std::string message(int v) const override { return "fake " + std::to_string(v); }
    std::error_condition default_error_condition(int v) const noexcept override {
        if (v == 5) return std::errc::permission_denied;
        if (v == 2) return std::errc::no_such_file_or_directory;
        return std::error_condition(v, *this);
    }
};
const FakeWin32 fake_win32;

void test_errno_of() {
    check::equal(util::errno_of(std::error_code()), EIO, "no error: EIO");
    check::equal(util::errno_of(std::make_error_code(std::errc::no_such_file_or_directory)), ENOENT, "generic");
    check::equal(util::errno_of(std::error_code(5, fake_win32)), EACCES, "ERROR_ACCESS_DENIED -> EACCES, not 5");
    check::equal(util::errno_of(std::error_code(2, fake_win32)), ENOENT, "ERROR_FILE_NOT_FOUND -> ENOENT");
    check::equal(util::errno_of(std::error_code(1234, fake_win32)), EIO, "no errno: EIO");
#ifndef _WIN32
    check::equal(util::errno_of(std::error_code(ENOSPC, std::system_category())), ENOSPC, "POSIX system: errno");
    check::equal(util::errno_of(std::error_code(4000, std::system_category())), 4000, "POSIX system: as it is");
#endif
    const std::error_code c = util::errno_code_of(std::error_code(5, fake_win32));
    check::is_true(c.category() == std::generic_category() && c.value() == EACCES, "errno_code_of: generic EACCES");
}

void test_replace_file() {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / ("sstvae_fileio_" + std::to_string(std::rand()));
    fs::create_directories(dir);
    const std::string a = util::path_text(dir / "a.tmp"), b = util::path_text(dir / "b");
    for (const auto& [p, text] : {std::pair{a, "new"}, std::pair{b, "old"}}) {
        util::File f = util::open_file(p, "wb");
        std::fputs(text, f.get());
    }
    std::error_code ec;
    check::is_true(util::replace_file(a, b, ec) && !ec, "replaces an existing target");
    check::is_true(!util::exists(a) && util::is_file(b), "moved");
    {
        util::File f = util::open_file(b, "rb");
        char buf[8] = {};
        check::equal(std::string(buf, std::fread(buf, 1, sizeof buf, f.get())), std::string("new"), "the new text");
    }
    check::is_true(!util::replace_file(a, b, ec), "a missing source fails");
    check::equal(util::errno_of(ec), ENOENT, "ENOENT");
    check::is_true(!util::replace_file(std::string("x\0y", 3), b, ec), "a NUL in the name");
    check::equal(util::errno_of(ec), EINVAL, "EINVAL");
    fs::remove_all(dir);
}

}  // namespace

int main() {
    try {
        test_errno_of();
        test_replace_file();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return check::report("qrss_core_fileio_errno");
}
