#include "qrss/pynum.hpp"

#include <clocale>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "qrss/pytext.hpp"
#include "qrss/pyunicode_tables.hpp"

namespace sstvae::qrss::pynum {

namespace {

// The whitespace int() and float() strip: Py_ISSPACE, 0x09-0x0D and the
// space. Not str.isspace(), which also counts 0x1C-0x1F: the two map only
// characters past ASCII to ' ' (_PyUnicode_TransformDecimalAndSpaceToASCII)
// and then strip with Py_ISSPACE, so int("\x1c5") is a ValueError.
bool py_space(char c) {
    const auto u = static_cast<unsigned char>(c);
    return u == 0x20 || (u >= 0x09 && u <= 0x0D);
}

// One code point of strict UTF-8 at s[i] (RFC 3629, no surrogates), i moved
// past it; nullopt for anything else.
std::optional<char32_t> next_code_point(std::string_view s, std::size_t& i) {
    const auto b = static_cast<unsigned char>(s[i]);
    if (b < 0x80) {
        ++i;
        return b;
    }
    std::size_t len = 0;
    unsigned char lo = 0x80, hi = 0xBF;
    if (b >= 0xC2 && b <= 0xDF) {
        len = 2;
    } else if (b >= 0xE0 && b <= 0xEF) {
        len = 3;
        if (b == 0xE0) lo = 0xA0;
        if (b == 0xED) hi = 0x9F;   // not the surrogates
    } else if (b >= 0xF0 && b <= 0xF4) {
        len = 4;
        if (b == 0xF0) lo = 0x90;
        if (b == 0xF4) hi = 0x8F;
    } else {
        return std::nullopt;
    }
    if (i + len > s.size()) return std::nullopt;
    const auto b1 = static_cast<unsigned char>(s[i + 1]);
    if (b1 < lo || b1 > hi) return std::nullopt;
    char32_t cp = b & (len == 2 ? 0x1Fu : len == 3 ? 0x0Fu : 0x07u);
    for (std::size_t k = 1; k < len; ++k) {
        const auto c = static_cast<unsigned char>(s[i + k]);
        if ((c & 0xC0) != 0x80) return std::nullopt;
        cp = (cp << 6) | (c & 0x3Fu);
    }
    i += len;
    return cp;
}

bool unicode_space(char32_t cp) {
    for (const std::uint32_t c : pyunicode::SPACES)
        if (c == cp) return true;
    return false;
}

int unicode_digit(char32_t cp) {
    for (const pyunicode::DigitRun& r : pyunicode::DIGITS)
        if (cp >= r.first && cp <= r.last) return r.value + static_cast<int>(cp - r.first);
    return -1;
}

// The text int() and float() parse: _PyUnicode_TransformDecimalAndSpaceToASCII
// of it -- ASCII kept, a Unicode space (Py_UNICODE_ISSPACE) read as ' ', a
// Unicode decimal digit as its ASCII digit; anything else past ASCII (DEL
// included, and a byte that is not UTF-8: a surrogate escape there) is the
// reference's '?', which no number holds, so nullopt.
std::optional<std::string> to_ascii(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    std::size_t i = 0;
    while (i < text.size()) {
        const std::optional<char32_t> cp = next_code_point(text, i);
        if (!cp) return std::nullopt;
        if (*cp < 127) {
            out += static_cast<char>(*cp);
        } else if (unicode_space(*cp)) {
            out += ' ';
        } else if (const int d = unicode_digit(*cp); d >= 0) {
            out += static_cast<char>('0' + d);
        } else {
            return std::nullopt;
        }
    }
    return out;
}

std::string_view strip(std::string_view s) {
    std::size_t i = 0, n = s.size();
    while (i < n && py_space(s[i])) ++i;
    while (n > i && py_space(s[n - 1])) --n;
    return s.substr(i, n - i);
}

int digit_value(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'z') return c - 'a' + 10;
    if (c >= 'A' && c <= 'Z') return c - 'A' + 10;
    return 99;
}

bool is_digit(char c) { return c >= '0' && c <= '9'; }

char lower(char c) { return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c; }

bool iequals(std::string_view a, std::string_view b) {
    if (a.size() != b.size()) return false;
    for (std::size_t k = 0; k < a.size(); ++k)
        if (lower(a[k]) != b[k]) return false;
    return true;
}

// UTF-8 of one code point (repr() output only ever holds valid ones).
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

}  // namespace

std::optional<PyInt> py_int(std::string_view text, int base) {
    if (base != 10 && base != 16) throw std::invalid_argument("py_int: base must be 10 or 16");
    const std::optional<std::string> ascii = to_ascii(text);
    if (!ascii) return std::nullopt;
    const std::string_view s = strip(*ascii);
    std::size_t i = 0;
    const std::size_t n = s.size();
    bool negative = false;
    if (i < n && (s[i] == '+' || s[i] == '-')) negative = s[i++] == '-';
    if (base == 16 && i + 1 < n && s[i] == '0' && (s[i + 1] == 'x' || s[i + 1] == 'X')) {
        i += 2;
        if (i < n && s[i] == '_') ++i;   // one underscore may follow the prefix
    }
    std::vector<int> digits;
    bool after_underscore = false;
    for (; i < n; ++i) {
        const char c = s[i];
        if (c == '_') {
            if (digits.empty() || after_underscore) return std::nullopt;
            after_underscore = true;
            continue;
        }
        const int d = digit_value(c);
        if (d >= base) return std::nullopt;
        digits.push_back(d);
        after_underscore = false;
    }
    if (digits.empty() || after_underscore) return std::nullopt;

    // The value in decimal, schoolbook: Python's int has no limit.
    std::vector<int> dec{0};   // little-endian decimal digits
    for (const int d : digits) {
        int carry = d;
        for (int& x : dec) {
            const int v = x * base + carry;
            x = v % 10;
            carry = v / 10;
        }
        while (carry) {
            dec.push_back(carry % 10);
            carry /= 10;
        }
    }
    while (dec.size() > 1 && dec.back() == 0) dec.pop_back();
    const bool zero = dec.size() == 1 && dec[0] == 0;
    PyInt out;
    if (negative && !zero) out.decimal = "-";
    for (auto it = dec.rbegin(); it != dec.rend(); ++it) out.decimal += static_cast<char>('0' + *it);
    // int64 when it fits: |v| <= 2^63 - 1, or exactly -2^63.
    std::uint64_t mag = 0;
    bool fits = dec.size() <= 19;
    if (fits) {
        for (auto it = dec.rbegin(); it != dec.rend(); ++it) mag = mag * 10 + static_cast<unsigned>(*it);
        fits = negative ? mag <= 9223372036854775808ULL : mag <= 9223372036854775807ULL;
    }
    if (fits) {
        if (!negative) out.value = static_cast<std::int64_t>(mag);
        else out.value = mag == 9223372036854775808ULL ? std::numeric_limits<std::int64_t>::min()
                                                       : -static_cast<std::int64_t>(mag);
    }
    return out;
}

std::string invalid_int_literal(std::string_view text, int base) {
    // "%.200R": the repr, cut to 200 characters.
    const std::u32string cps = pytext::decode(pytext::repr(text));
    std::string shown;
    for (std::size_t k = 0; k < cps.size() && k < 200; ++k) put_utf8(shown, cps[k]);
    return "invalid literal for int() with base " + std::to_string(base) + ": " + shown;
}

std::optional<double> py_float(std::string_view text) {
    const std::optional<std::string> ascii = to_ascii(text);
    if (!ascii) return std::nullopt;
    const std::string_view s = strip(*ascii);
    if (s.empty()) return std::nullopt;

    // Underscores only between two digits (_Py_string_to_number_with_underscores).
    std::string t;
    t.reserve(s.size());
    char prev = '\0';
    for (const char c : s) {
        if (c == '_') {
            if (!is_digit(prev)) return std::nullopt;
        } else {
            if (prev == '_' && !is_digit(c)) return std::nullopt;
            t += c;
        }
        prev = c;
    }
    if (prev == '_') return std::nullopt;

    std::size_t i = 0;
    bool negative = false;
    if (t[i] == '+' || t[i] == '-') negative = t[i++] == '-';
    const std::string_view body = std::string_view(t).substr(i);
    if (iequals(body, "inf") || iequals(body, "infinity"))
        return negative ? -std::numeric_limits<double>::infinity() : std::numeric_limits<double>::infinity();
    if (iequals(body, "nan"))
        return std::copysign(std::numeric_limits<double>::quiet_NaN(), negative ? -1.0 : 1.0);

    // digits [. digits] [e [sign] digits], with at least one mantissa digit.
    std::size_t k = 0, mantissa = 0;
    while (k < body.size() && is_digit(body[k])) ++k, ++mantissa;
    if (k < body.size() && body[k] == '.') {
        ++k;
        while (k < body.size() && is_digit(body[k])) ++k, ++mantissa;
    }
    if (mantissa == 0) return std::nullopt;
    if (k < body.size() && (body[k] == 'e' || body[k] == 'E')) {
        ++k;
        if (k < body.size() && (body[k] == '+' || body[k] == '-')) ++k;
        std::size_t exp_digits = 0;
        while (k < body.size() && is_digit(body[k])) ++k, ++exp_digits;
        if (exp_digits == 0) return std::nullopt;
    }
    if (k != body.size()) return std::nullopt;
    // Validated, so strtod reads the whole of it (its hex, inf and nan forms
    // never reach here); it rounds correctly and overflows to +-inf as float().
    // strtod's decimal point is LC_NUMERIC's, which a Qt program has set from
    // the environment ("1,5" in de_DE): spell the point as the locale does.
    const char* point = std::localeconv()->decimal_point;
    if (point && std::strcmp(point, ".") != 0 && *point) {
        const std::size_t dot = t.find('.');
        if (dot != std::string::npos) t.replace(dot, 1, point);
    }
    return std::strtod(t.c_str(), nullptr);
}

}  // namespace sstvae::qrss::pynum
