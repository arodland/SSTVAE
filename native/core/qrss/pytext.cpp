#include "qrss/pytext.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace sstvae::qrss::pytext {

namespace {

// Code points >= 0x80 for which str.isprintable() is False and the
// Unicode category is not Cn (unassigned): Cc, Cf, Cs, Co, Zs, Zl, Zp.
// From Python 3.13 (Unicode 15.1):
//   [c for c in range(0x80, 0x110000) if not chr(c).isprintable()
//    and unicodedata.category(chr(c)) != "Cn"], as inclusive ranges.
struct Range {
    char32_t lo, hi;
};
constexpr std::array<Range, 27> NOT_PRINTABLE{{
    {0x80, 0xa0},       {0xad, 0xad},       {0x600, 0x605},     {0x61c, 0x61c},
    {0x6dd, 0x6dd},     {0x70f, 0x70f},     {0x890, 0x891},     {0x8e2, 0x8e2},
    {0x1680, 0x1680},   {0x180e, 0x180e},   {0x2000, 0x200f},   {0x2028, 0x202f},
    {0x205f, 0x2064},   {0x2066, 0x206f},   {0x3000, 0x3000},   {0xd800, 0xf8ff},
    {0xfeff, 0xfeff},   {0xfff9, 0xfffb},   {0x110bd, 0x110bd}, {0x110cd, 0x110cd},
    {0x13430, 0x1343f}, {0x1bca0, 0x1bca3}, {0x1d173, 0x1d17a}, {0xe0001, 0xe0001},
    {0xe0020, 0xe007f}, {0xf0000, 0xffffd}, {0x100000, 0x10fffd},
}};

constexpr char HEX[] = "0123456789abcdef";

bool cont(unsigned char b) { return (b & 0xC0u) == 0x80u; }

void put_utf8(std::string& out, char32_t cp) {
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
}

void put_hex(std::string& out, char32_t cp, int digits) {
    for (int k = digits - 1; k >= 0; --k) out += HEX[(cp >> (4 * k)) & 0xF];
}

}  // namespace

std::u32string decode(std::string_view utf8) {
    std::u32string out;
    out.reserve(utf8.size());
    const auto* s = reinterpret_cast<const unsigned char*>(utf8.data());
    const std::size_t n = utf8.size();
    std::size_t i = 0;
    while (i < n) {
        const unsigned char b = s[i];
        if (b < 0x80) {
            out += static_cast<char32_t>(b);
            ++i;
            continue;
        }
        // The legal range of the second byte depends on the first (RFC 3629),
        // except that ED A0..BF (a surrogate) is accepted, as "surrogatepass".
        std::size_t len = 0;
        unsigned char lo = 0x80, hi = 0xBF;
        if (b >= 0xC2 && b <= 0xDF) {
            len = 2;
        } else if (b >= 0xE0 && b <= 0xEF) {
            len = 3;
            if (b == 0xE0) lo = 0xA0;
        } else if (b >= 0xF0 && b <= 0xF4) {
            len = 4;
            if (b == 0xF0) lo = 0x90;
            if (b == 0xF4) hi = 0x8F;
        }
        bool ok = len != 0 && i + len <= n && s[i + 1] >= lo && s[i + 1] <= hi;
        for (std::size_t k = 2; ok && k < len; ++k) ok = cont(s[i + k]);
        if (!ok) {
            out += static_cast<char32_t>(0xDC00 + b);   // surrogateescape
            ++i;
            continue;
        }
        char32_t cp = b & (len == 2 ? 0x1F : len == 3 ? 0x0F : 0x07);
        for (std::size_t k = 1; k < len; ++k) cp = (cp << 6) | (s[i + k] & 0x3F);
        out += cp;
        i += len;
    }
    return out;
}

bool isprintable(char32_t cp) {
    if (cp < 0x80) return cp >= 0x20 && cp < 0x7F;
    for (const Range& r : NOT_PRINTABLE) {
        if (cp < r.lo) return true;
        if (cp <= r.hi) return false;
    }
    return true;
}

std::string repr(std::u32string_view text) {
    // CPython's unicode_repr: single quotes unless the text has a single
    // quote and no double quote.
    bool squote = false, dquote = false;
    for (char32_t c : text) {
        squote = squote || c == U'\'';
        dquote = dquote || c == U'"';
    }
    const char32_t quote = (squote && !dquote) ? U'"' : U'\'';
    std::string out;
    out += static_cast<char>(quote);
    for (char32_t ch : text) {
        if (ch == quote || ch == U'\\') {
            out += '\\';
            out += static_cast<char>(ch);
        } else if (ch == U'\t') {
            out += "\\t";
        } else if (ch == U'\n') {
            out += "\\n";
        } else if (ch == U'\r') {
            out += "\\r";
        } else if (ch < 0x20 || ch == 0x7F) {
            out += "\\x";
            put_hex(out, ch, 2);
        } else if (ch < 0x7F) {
            out += static_cast<char>(ch);
        } else if (!isprintable(ch)) {
            if (ch <= 0xFF) {
                out += "\\x";
                put_hex(out, ch, 2);
            } else if (ch <= 0xFFFF) {
                out += "\\u";
                put_hex(out, ch, 4);
            } else {
                out += "\\U";
                put_hex(out, ch, 8);
            }
        } else {
            put_utf8(out, ch);
        }
    }
    out += static_cast<char>(quote);
    return out;
}

}  // namespace sstvae::qrss::pytext
