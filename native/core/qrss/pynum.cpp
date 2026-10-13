#include "qrss/pynum.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "qrss/pytext.hpp"

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
    const std::string_view s = strip(text);
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
    const std::string_view s = strip(text);
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
    return std::strtod(t.c_str(), nullptr);
}

}  // namespace sstvae::qrss::pynum
