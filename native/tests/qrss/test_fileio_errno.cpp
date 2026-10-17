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
#include "qrss/rawfile.hpp"
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

// rawfile::FileError (the passband store's, the FLAC packer's, the directory
// lock's OSError) carries an errno whatever it is made from: a Win32 code
// reaching a binding as 5 would be an OSError saying EIO for an access denied.
void test_file_error() {
    using qrss::rawfile::FileError;
    const FileError a(std::error_code(5, fake_win32), "/x/fe_1.i16");
    check::is_true(a.code().category() == std::generic_category(), "a Win32-style code becomes an errno");
    check::equal(a.code().value(), EACCES, "ERROR_ACCESS_DENIED -> EACCES");
    check::equal(a.path(), std::string("/x/fe_1.i16"), "the path");
    check::is_true(a.code() == std::errc::permission_denied, "compares as EACCES");
    const FileError b(std::error_code(2, fake_win32), "p");
    check::is_true(b.code() == std::errc::no_such_file_or_directory, "ERROR_FILE_NOT_FOUND -> ENOENT (is_missing)");
    check::equal(FileError(std::error_code(1234, fake_win32), "p").code().value(), EIO, "no errno: EIO");
    check::equal(FileError(std::make_error_code(std::errc::no_space_on_device), "p").code().value(), ENOSPC,
                 "an errno stays");
    check::equal(FileError(std::error_code(), "p").code().value(), EIO, "no error at all: EIO, never 0");
#ifndef _WIN32
    check::equal(FileError(std::error_code(EROFS, std::system_category()), "p").code().value(), EROFS,
                 "POSIX system_category: its errno");
#endif
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
        test_file_error();
        test_replace_file();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return check::report("qrss_core_fileio_errno");
}
