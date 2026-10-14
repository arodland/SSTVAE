// File names in the native code: UTF-8 in a std::string, everywhere, and
// every file opened through this header.
//
// On POSIX the bytes go to the system as they are: a file name is bytes
// there, so a name that is not UTF-8 at all (what Python hands over as
// os.fsencode of a surrogateescape'd str, or as bytes) opens like any
// other. On Windows they are converted to UTF-16 and opened through the
// wide API (_wfopen, std::filesystem's wchar_t paths). std::fopen and
// std::ifstream(std::string) read a narrow name through the ANSI code
// page instead, so a name outside it (Cyrillic on a Western system, say)
// cannot be opened at all, and a UTF-8 name inside it opens a different
// file. Paths reach the C++ as UTF-8 from both directions: the bindings
// encode a str with the file-system encoding (UTF-8 on Windows), and
// sstvae-qrss converts its wide command line (apps/sstvae_qrss/main.cpp).
//
// The conversion is Python's own on Windows (PEP 529: file names are
// "utf-8" with "surrogatepass"). An NTFS name may hold a lone surrogate;
// it travels as its three-byte encoding, so every Windows name has
// exactly one byte string and back. It is written out here rather than
// taken from MultiByteToWideChar, which turns those three bytes into
// U+FFFD (a name Python can open would not open here), and so that it is
// tested on every platform (native/tests/qrss/test_paths.cpp) rather than
// only on the one CI machine that runs Windows.
//
// Header-only and Win32-header-free: _wfopen comes from <cstdio> on both
// Windows toolchains, and nothing here needs <windows.h> (CLAUDE.md:
// never from a widely-included header). Header-only because sstvae_core
// lists its sources by hand while this is wanted by sstvae_core's and
// sstvae_qrss's code alike.

#ifndef SSTVAE_UTIL_FILEIO_HPP
#define SSTVAE_UTIL_FILEIO_HPP

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace sstvae::util {

// --- UTF-8 <-> UTF-16, as Python converts Windows file names ----------------------

// UTF-8 to UTF-16. Well-formed UTF-8 (RFC 3629: no overlong forms, nothing
// past U+10FFFF) plus a surrogate's three-byte form (ED A0..BF 80..BF, what
// "surrogatepass" writes), which becomes that one code unit. nullopt for
// anything else, where Python raises UnicodeDecodeError.
inline std::optional<std::u16string> utf16_of(std::string_view utf8) {
    std::u16string out;
    out.reserve(utf8.size());
    const auto* s = reinterpret_cast<const unsigned char*>(utf8.data());
    const std::size_t n = utf8.size();
    auto cont = [&](std::size_t k) { return k < n && (s[k] & 0xC0) == 0x80; };
    std::size_t i = 0;
    while (i < n) {
        const unsigned char b = s[i];
        if (b < 0x80) {
            out += static_cast<char16_t>(b);
            ++i;
            continue;
        }
        std::size_t len = 0;
        unsigned char lo = 0x80, hi = 0xBF;   // the legal second bytes
        if (b >= 0xC2 && b <= 0xDF) {
            len = 2;
        } else if (b >= 0xE0 && b <= 0xEF) {
            len = 3;
            if (b == 0xE0) lo = 0xA0;   // ED's A0..BF are the surrogates: allowed
        } else if (b >= 0xF0 && b <= 0xF4) {
            len = 4;
            if (b == 0xF0) lo = 0x90;
            if (b == 0xF4) hi = 0x8F;
        } else {
            return std::nullopt;
        }
        if (i + 1 >= n || s[i + 1] < lo || s[i + 1] > hi) return std::nullopt;
        for (std::size_t k = 2; k < len; ++k)
            if (!cont(i + k)) return std::nullopt;
        std::uint32_t cp = b & (len == 2 ? 0x1Fu : len == 3 ? 0x0Fu : 0x07u);
        for (std::size_t k = 1; k < len; ++k) cp = (cp << 6) | (s[i + k] & 0x3Fu);
        if (cp >= 0x10000) {
            cp -= 0x10000;
            out += static_cast<char16_t>(0xD800 + (cp >> 10));
            out += static_cast<char16_t>(0xDC00 + (cp & 0x3FF));
        } else {
            out += static_cast<char16_t>(cp);
        }
        i += len;
    }
    return out;
}

// UTF-16 to UTF-8: a high surrogate followed by a low one is one code point
// (four bytes), a lone surrogate its own three bytes. Total: every UTF-16
// string has one, and utf16_of gives it back.
inline std::string utf8_of(std::u16string_view utf16) {
    std::string out;
    out.reserve(utf16.size());
    auto put = [&out](std::uint32_t cp) {
        if (cp < 0x80) {
            out += static_cast<char>(cp);
        } else if (cp < 0x800) {
            out += static_cast<char>(0xC0 | (cp >> 6));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            out += static_cast<char>(0xE0 | (cp >> 12));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else {
            out += static_cast<char>(0xF0 | (cp >> 18));
            out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        }
    };
    for (std::size_t i = 0; i < utf16.size(); ++i) {
        const std::uint32_t u = utf16[i];
        if (u >= 0xD800 && u < 0xDC00 && i + 1 < utf16.size() && utf16[i + 1] >= 0xDC00 && utf16[i + 1] < 0xE000) {
            put(0x10000 + ((u - 0xD800) << 10) + (static_cast<std::uint32_t>(utf16[i + 1]) - 0xDC00));
            ++i;
        } else {
            put(u);
        }
    }
    return out;
}

#ifdef _WIN32
// A UTF-8 name as the wide string the Windows API takes (wchar_t is UTF-16
// there); nullopt when it is not UTF-8.
inline std::optional<std::wstring> wide_of(std::string_view utf8) {
    const std::optional<std::u16string> u = utf16_of(utf8);
    if (!u) return std::nullopt;
    return std::wstring(u->begin(), u->end());
}

inline std::string utf8_of(std::wstring_view wide) {
    return utf8_of(std::u16string(wide.begin(), wide.end()));
}
#endif

// --- opening a file --------------------------------------------------------------

struct FileCloser {
    void operator()(std::FILE* f) const noexcept {
        if (f) std::fclose(f);
    }
};
using File = std::unique_ptr<std::FILE, FileCloser>;

// std::fopen(path, mode) for a UTF-8 path. Null with errno set when it
// fails: the system's errno, EINVAL for a name holding a NUL byte (fopen
// would open the name before it; Python refuses "embedded null byte"
// before it opens anything), and on Windows EILSEQ for one that is not
// UTF-8.
inline File open_file(const std::string& path, const char* mode) {
    if (path.find('\0') != std::string::npos) {
        errno = EINVAL;
        return File();
    }
#ifdef _WIN32
    const std::optional<std::wstring> wpath = wide_of(path);
    if (!wpath) {
        errno = EILSEQ;
        return File();
    }
    std::wstring wmode;
    for (const char* m = mode; *m; ++m) wmode += static_cast<wchar_t>(static_cast<unsigned char>(*m));
    return File(::_wfopen(wpath->c_str(), wmode.c_str()));
#else
    return File(std::fopen(path.c_str(), mode));
#endif
}

// --- std::filesystem -----------------------------------------------------------------

// The std::filesystem::path a UTF-8 name means (its bytes on POSIX, its
// UTF-16 on Windows); nullopt for a name holding a NUL byte or, on Windows,
// one that is not UTF-8.
inline std::optional<std::filesystem::path> fs_path(const std::string& path) {
    if (path.find('\0') != std::string::npos) return std::nullopt;
#ifdef _WIN32
    const std::optional<std::wstring> w = wide_of(path);
    if (!w) return std::nullopt;
    return std::filesystem::path(*w);
#else
    return std::filesystem::path(path);
#endif
}

// A path's name as UTF-8 (its bytes on POSIX). Never path::string(), which
// goes through the ANSI code page on Windows.
inline std::string path_text(const std::filesystem::path& p) {
#ifdef _WIN32
    return utf8_of(std::wstring_view(p.native()));
#else
    return p.native();
#endif
}

// os.path.exists / os.path.isfile / os.path.isdir: symlinks followed, and
// false on any error, a name holding a NUL byte included (Python's
// ValueError is caught there too).
inline bool exists(const std::string& path) {
    const auto p = fs_path(path);
    std::error_code ec;
    return p && std::filesystem::exists(*p, ec);
}

inline bool is_file(const std::string& path) {
    const auto p = fs_path(path);
    std::error_code ec;
    return p && std::filesystem::is_regular_file(*p, ec);
}

inline bool is_dir(const std::string& path) {
    const auto p = fs_path(path);
    std::error_code ec;
    return p && std::filesystem::is_directory(*p, ec);
}

// The entry names of a directory, as UTF-8 and in the system's order
// (os.listdir: no "." or ".."); `ec` set, and nothing returned, when it
// cannot be read.
inline std::vector<std::string> list_dir(const std::string& dir, std::error_code& ec) {
    ec.clear();
    std::vector<std::string> out;
    const auto p = fs_path(dir);
    if (!p) {
        ec = std::make_error_code(std::errc::invalid_argument);
        return out;
    }
    std::filesystem::directory_iterator it(*p, ec), end;
    for (; !ec && it != end; it.increment(ec)) out.push_back(path_text(it->path().filename()));
    if (ec) out.clear();
    return out;
}

// os.remove (a file, not a directory) and os.replace (a rename that
// replaces an existing target): true on success, `ec` set otherwise.
inline bool remove_file(const std::string& path, std::error_code& ec) {
    ec.clear();
    const auto p = fs_path(path);
    if (!p) {
        ec = std::make_error_code(std::errc::invalid_argument);
        return false;
    }
    if (std::filesystem::is_directory(*p, ec)) {
        ec = std::make_error_code(std::errc::is_a_directory);
        return false;
    }
    ec.clear();
    if (!std::filesystem::remove(*p, ec) && !ec) ec = std::make_error_code(std::errc::no_such_file_or_directory);
    return !ec;
}

inline bool replace_file(const std::string& from, const std::string& to, std::error_code& ec) {
    ec.clear();
    const auto a = fs_path(from);
    const auto b = fs_path(to);
    if (!a || !b) {
        ec = std::make_error_code(std::errc::invalid_argument);
        return false;
    }
    std::filesystem::rename(*a, *b, ec);
    return !ec;
}

}  // namespace sstvae::util

#endif  // SSTVAE_UTIL_FILEIO_HPP
