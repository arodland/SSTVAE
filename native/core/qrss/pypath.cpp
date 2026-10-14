#include "qrss/pypath.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

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
    if (parts.empty()) throw std::invalid_argument(pathlib_str(p) + " has an empty name");
    parts.back() = name;
    return join(root, parts);
}

std::size_t stderr_text(std::string_view text, bool final, std::string& out) {
    static constexpr char HEX[] = "0123456789abcdef";
    const auto escape = [&out](unsigned cp) {
        out += "\\u";
        for (int shift = 12; shift >= 0; shift -= 4) out += HEX[(cp >> shift) & 0xF];
    };
#ifdef _WIN32
    constexpr bool surrogates = true;
#else
    constexpr bool surrogates = false;
#endif
    const auto byte = [text](std::size_t k) { return static_cast<unsigned char>(text[k]); };
    const std::size_t n = text.size();
    std::size_t i = 0;
    while (i < n) {
        const unsigned b = byte(i);
        if (b < 0x80) {
            out += static_cast<char>(b);
            ++i;
            continue;
        }
        // The sequence b leads, as Python's strict UTF-8 decoder bounds its
        // second byte: no overlong form, no surrogate (but on Windows), no
        // code point past U+10FFFF.
        std::size_t len = 0;
        unsigned lo = 0x80, hi = 0xBF;
        if (b >= 0xC2 && b <= 0xDF) {
            len = 2;
        } else if (b >= 0xE0 && b <= 0xEF) {
            len = 3;
            if (b == 0xE0) lo = 0xA0;
            if (b == 0xED && !surrogates) hi = 0x9F;
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
        if (!bad && k < len) {          // cut short by the end of text
            if (!final) return i;
            bad = true;
        }
        if (bad) {                      // this byte alone; the rest are read again
            escape(0xDC00 + b);
            ++i;
            continue;
        }
        if (len == 3 && b == 0xED && byte(i + 1) >= 0xA0) {   // a lone surrogate (Windows)
            escape(((b & 0x0Fu) << 12) | ((byte(i + 1) & 0x3Fu) << 6) | (byte(i + 2) & 0x3Fu));
        } else {
            out.append(text.substr(i, len));
        }
        i += len;
    }
    return i;
}

}  // namespace sstvae::qrss::pypath
