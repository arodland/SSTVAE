#include "qrss/pypath.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "qrss/pytext.hpp"

namespace sstvae::qrss::pypath {

std::pair<std::string, std::vector<std::string>> split(std::string_view text) {
    std::string root;
    std::string_view rest = text;
    if (!rest.empty() && rest[0] == '/') {
        // POSIX keeps exactly two leading slashes (implementation-defined
        // there); one, or three and more, are one.
        if (rest.size() >= 2 && rest[1] == '/' && !(rest.size() >= 3 && rest[2] == '/')) {
            root = "//";
            rest.remove_prefix(2);
        } else {
            root = "/";
            rest.remove_prefix(1);
        }
    }
    std::vector<std::string> parts;
    std::size_t i = 0;
    while (i <= rest.size()) {
        const std::size_t j = std::min(rest.find('/', i), rest.size());
        const std::string_view part = rest.substr(i, j - i);
        if (!part.empty() && part != ".") parts.emplace_back(part);
        i = j + 1;
    }
    return {root, parts};
}

std::string join(const std::string& root, const std::vector<std::string>& parts) {
    std::string out = root;
    for (std::size_t k = 0; k < parts.size(); ++k) {
        if (k) out += '/';
        out += parts[k];
    }
    return out.empty() ? "." : out;
}

std::string pathlib_str(std::string_view text) {
    const auto [root, parts] = split(text);
    return join(root, parts);
}

std::string pathlib_name(std::string_view text) {
    const auto [root, parts] = split(text);
    return parts.empty() ? std::string() : parts.back();
}

std::string pathlib_suffix(std::string_view text) {
    const std::string name = pathlib_name(text);
    const std::size_t dot = name.rfind('.');
    // 0 < i < len(name) - 1: a leading dot is not a suffix, nor a trailing one.
    if (dot == std::string::npos || dot == 0 || dot + 1 == name.size()) return {};
    return name.substr(dot);
}

std::string child(std::string_view dir, const std::string& name) {
    auto [root, parts] = split(dir);
    parts.push_back(name);
    return join(root, parts);
}

std::string with_name(std::string_view p, const std::string& name) {
    auto [root, parts] = split(p);
    // Path(p).with_name's ValueError, naming the path by its repr (POSIX
    // spelling: PosixPath, as this file spells every path)
    if (parts.empty()) throw std::invalid_argument("PosixPath(" + pytext::repr(fs_decode(pathlib_str(p))) +
                                                   ") has an empty name");
    parts.back() = name;
    return join(root, parts);
}

namespace {

#ifdef _WIN32
constexpr bool SURROGATES = true;    // the file-system codec is "surrogatepass" (PEP 529)
#else
constexpr bool SURROGATES = false;   // "surrogateescape" only
#endif

// One code point of os.fsdecode(text) at text[i], and the bytes it took:
// Python's strict UTF-8 decoder (no overlong form, no code point past
// U+10FFFF, no surrogate but on Windows), and a byte it refuses as U+DC00 +
// that byte alone (surrogateescape), the bytes after it read again. len 0:
// a sequence the end of `text` cuts short, when not `final`.
struct Unit {
    char32_t cp;
    std::size_t len;
};

Unit fs_unit(std::string_view text, std::size_t i, bool final) {
    const auto byte = [text](std::size_t k) { return static_cast<unsigned char>(text[k]); };
    const std::size_t n = text.size();
    const unsigned b = byte(i);
    if (b < 0x80) return {static_cast<char32_t>(b), 1};
    // The sequence b leads, as Python's decoder bounds its second byte.
    std::size_t len = 0;
    unsigned lo = 0x80, hi = 0xBF;
    if (b >= 0xC2 && b <= 0xDF) {
        len = 2;
    } else if (b >= 0xE0 && b <= 0xEF) {
        len = 3;
        if (b == 0xE0) lo = 0xA0;
        if (b == 0xED && !SURROGATES) hi = 0x9F;
    } else if (b >= 0xF0 && b <= 0xF4) {
        len = 4;
        if (b == 0xF0) lo = 0x90;
        if (b == 0xF4) hi = 0x8F;
    }
    std::size_t k = 1;
    bool bad = len == 0;
    for (; !bad && k < len && i + k < n; ++k) {
        const unsigned c = byte(i + k);
        bad = k == 1 ? (c < lo || c > hi) : (c < 0x80 || c > 0xBF);
    }
    if (!bad && k < len) {   // cut short by the end of text
        if (!final) return {0, 0};
        bad = true;
    }
    if (bad) return {static_cast<char32_t>(0xDC00 + b), 1};
    std::uint32_t cp = b & (len == 2 ? 0x1Fu : len == 3 ? 0x0Fu : 0x07u);
    for (k = 1; k < len; ++k) cp = (cp << 6) | (byte(i + k) & 0x3Fu);
    return {static_cast<char32_t>(cp), len};
}

}  // namespace

std::u32string fs_decode(std::string_view text) {
    std::u32string out;
    for (std::size_t i = 0; i < text.size();) {
        const Unit u = fs_unit(text, i, true);
        out += u.cp;
        i += u.len;
    }
    return out;
}

std::string fs_str(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    for (std::size_t i = 0; i < text.size();) {
        const Unit u = fs_unit(text, i, true);
        if (u.cp >= 0xD800 && u.cp < 0xE000) {
            out += static_cast<char>(0xE0 | (u.cp >> 12));
            out += static_cast<char>(0x80 | ((u.cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (u.cp & 0x3F));
        } else {
            out.append(text.substr(i, u.len));
        }
        i += u.len;
    }
    return out;
}

std::size_t stderr_text(std::string_view text, bool final, std::string& out) {
    static constexpr char HEX[] = "0123456789abcdef";
    std::size_t i = 0;
    while (i < text.size()) {
        const Unit u = fs_unit(text, i, final);
        if (u.len == 0) return i;   // left for the next call
        if (u.cp >= 0xD800 && u.cp < 0xE000) {
            // a surrogate (an escaped byte, or on Windows a lone surrogate's
            // three bytes): "backslashreplace"
            out += "\\u";
            for (int shift = 12; shift >= 0; shift -= 4) out += HEX[(u.cp >> shift) & 0xF];
        } else {
            out.append(text.substr(i, u.len));   // UTF-8 as it is
        }
        i += u.len;
    }
    return i;
}

}  // namespace sstvae::qrss::pypath
