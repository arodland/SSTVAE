#include "qrss/rational.hpp"

#include <bit>
#include <cmath>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace sstvae::qrss {

namespace {

constexpr std::int64_t I64_MAX = std::numeric_limits<std::int64_t>::max();
constexpr std::int64_t I64_LOW = -I64_MAX;   // the most negative value kept

[[noreturn]] void overflow(const char* where) {
    throw std::overflow_error(std::string("Rational: result out of int64 range in ") + where);
}

std::uint64_t uabs(std::int64_t a) {
    return a < 0 ? static_cast<std::uint64_t>(-(a + 1)) + 1u : static_cast<std::uint64_t>(a);
}

std::int64_t add(std::int64_t a, std::int64_t b, const char* where) {
    if (b > 0 ? a > I64_MAX - b : a < I64_LOW - b) overflow(where);
    return a + b;
}

std::int64_t mul(std::int64_t a, std::int64_t b, const char* where) {
    const std::uint64_t ua = uabs(a), ub = uabs(b);
    if (ua != 0 && ub > static_cast<std::uint64_t>(I64_MAX) / ua) overflow(where);
    const auto p = static_cast<std::int64_t>(ua * ub);
    return ((a < 0) != (b < 0)) ? -p : p;
}

std::int64_t gcd(std::int64_t a, std::int64_t b) {
    return static_cast<std::int64_t>(std::gcd(uabs(a), uabs(b)));
}

// Python's a // b and a % b for b > 0.
std::int64_t floordiv(std::int64_t a, std::int64_t b) {
    std::int64_t q = a / b;
    if (a % b < 0) --q;
    return q;
}

std::int64_t floormod(std::int64_t a, std::int64_t b) {
    std::int64_t r = a % b;
    if (r < 0) r += b;
    return r;
}

void check_range(std::int64_t v) {
    if (v < I64_LOW) throw std::overflow_error("Rational: -2^63 is out of range");
}

// The assigned code points str.isprintable() rejects above ASCII (Unicode
// 15.1, CPython 3.13: Cc, Cf, Zs, Zl, Zp, Cs, Co). Unassigned ones are
// left out -- the one place repr() here can differ from Python's.
constexpr std::uint32_t NONPRINTABLE[][2] = {
    {0x80, 0xa0},       {0xad, 0xad},       {0x600, 0x605},     {0x61c, 0x61c},     {0x6dd, 0x6dd},
    {0x70f, 0x70f},     {0x890, 0x891},     {0x8e2, 0x8e2},     {0x1680, 0x1680},   {0x180e, 0x180e},
    {0x2000, 0x200f},   {0x2028, 0x202f},   {0x205f, 0x2064},   {0x2066, 0x206f},   {0x3000, 0x3000},
    {0xd800, 0xf8ff},   {0xfeff, 0xfeff},   {0xfff9, 0xfffb},   {0x110bd, 0x110bd}, {0x110cd, 0x110cd},
    {0x13430, 0x1343f}, {0x1bca0, 0x1bca3}, {0x1d173, 0x1d17a}, {0xe0001, 0xe0001}, {0xe0020, 0xe007f},
    {0xf0000, 0xffffd}, {0x100000, 0x10fffd},
};

// One UTF-8 sequence at s[i]: its code point and length, or length 0 when
// it is not well formed.
std::pair<std::uint32_t, std::size_t> utf8_at(std::string_view s, std::size_t i) {
    const auto b = [&](std::size_t k) { return static_cast<unsigned char>(s[i + k]); };
    const unsigned c = b(0);
    const std::size_t n = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xe ? 3 : (c >> 3) == 0x1e ? 4 : 0;
    if (n == 0 || i + n > s.size()) return {0, 0};
    std::uint32_t cp = n == 1 ? c : c & (0x7fu >> n);
    for (std::size_t k = 1; k < n; ++k) {
        if ((b(k) & 0xc0) != 0x80) return {0, 0};
        cp = (cp << 6) | (b(k) & 0x3fu);
    }
    return {cp, n};
}

// Python's repr() of a str (given as UTF-8).
std::string py_repr(std::string_view s) {
    const bool has_single = s.find('\'') != std::string_view::npos;
    const bool has_double = s.find('"') != std::string_view::npos;
    const char quote = (has_single && !has_double) ? '"' : '\'';
    static const char* hex = "0123456789abcdef";
    std::string out(1, quote);
    for (std::size_t i = 0; i < s.size();) {
        const auto [cp, len] = utf8_at(s, i);
        if (len > 1) {
            bool printable = true;
            for (const auto& r : NONPRINTABLE) printable = printable && !(cp >= r[0] && cp <= r[1]);
            if (printable) {
                out.append(s.substr(i, len));
            } else {
                const int digits = cp < 0x100 ? 2 : cp < 0x10000 ? 4 : 8;
                out += digits == 2 ? "\\x" : digits == 4 ? "\\u" : "\\U";
                for (int k = digits - 1; k >= 0; --k) out += hex[(cp >> (4 * k)) & 15u];
            }
            i += len;
            continue;
        }
        const char ch = s[i++];
        const auto c = static_cast<unsigned char>(ch);
        if (ch == quote || ch == '\\') {
            out += '\\';
            out += ch;
        } else if (ch == '\t') {
            out += "\\t";
        } else if (ch == '\n') {
            out += "\\n";
        } else if (ch == '\r') {
            out += "\\r";
        } else if (c < 0x20 || c == 0x7f) {
            out += "\\x";
            out += hex[c >> 4];
            out += hex[c & 15];
        } else {
            out += ch;
        }
    }
    out += quote;
    return out;
}

bool is_digit(char c) { return c >= '0' && c <= '9'; }

// Length of the whitespace character at s[i] (Python's str.isspace, as
// re's \s matches it, in UTF-8), or 0.
std::size_t space_at(std::string_view s, std::size_t i) {
    const auto b = [&](std::size_t k) { return i + k < s.size() ? static_cast<unsigned char>(s[i + k]) : 0u; };
    const unsigned c = b(0);
    if (c == ' ' || (c >= 0x09 && c <= 0x0d) || (c >= 0x1c && c <= 0x1f)) return 1;
    if (c == 0xc2 && (b(1) == 0x85 || b(1) == 0xa0)) return 2;                           // U+0085, U+00A0
    if (c == 0xe1 && b(1) == 0x9a && b(2) == 0x80) return 3;                            // U+1680
    if (c == 0xe2 && b(1) == 0x80 && (b(2) <= 0x8a || b(2) == 0xa8 || b(2) == 0xa9 || b(2) == 0xaf) &&
        b(2) >= 0x80)
        return 3;                                                                        // U+2000-200A, 2028, 2029, 202F
    if (c == 0xe2 && b(1) == 0x81 && b(2) == 0x9f) return 3;                            // U+205F
    if (c == 0xe3 && b(1) == 0x80 && b(2) == 0x80) return 3;                            // U+3000
    return 0;
}

void skip_space(std::string_view s, std::size_t& i) {
    while (std::size_t k = space_at(s, i)) i += k;
}

// \d+(_\d+)* at s[i], appended to `out` without the underscores. False
// (consuming nothing) when s[i] is not a digit.
bool digit_groups(std::string_view s, std::size_t& i, std::string& out) {
    if (!(i < s.size() && is_digit(s[i]))) return false;
    while (i < s.size() && is_digit(s[i])) out += s[i++];
    while (i + 1 < s.size() && s[i] == '_' && is_digit(s[i + 1])) {
        ++i;
        while (i < s.size() && is_digit(s[i])) out += s[i++];
    }
    return true;
}

// Decimal digits (no underscores) as int64; overflow_error past 2^63-1.
std::int64_t to_int64(std::string_view digits) {
    std::uint64_t v = 0;
    for (const char c : digits) {
        const auto d = static_cast<std::uint64_t>(c - '0');
        if (v > (static_cast<std::uint64_t>(I64_MAX) - d) / 10) overflow("Fraction(str)");
        v = v * 10 + d;
    }
    return static_cast<std::int64_t>(v);
}

std::int64_t pow_checked(std::int64_t base, std::int64_t k) {
    std::int64_t r = 1;
    for (std::int64_t i = 0; i < k; ++i) r = mul(r, base, "Fraction(str)");
    return r;
}

// A nonnegative integer of any width: what CPython's limit_denominator
// loop does to n and d when the float's exact denominator is 2^63 or
// more. Little-endian 32-bit limbs, no leading zero limb (zero is empty).
class Wide {
public:
    Wide() = default;
    explicit Wide(std::uint64_t v) {
        if (v != 0) l_.push_back(static_cast<std::uint32_t>(v));
        if ((v >> 32) != 0) l_.push_back(static_cast<std::uint32_t>(v >> 32));
    }
    static Wide pow2(int k) {
        Wide w;
        w.l_.assign(static_cast<std::size_t>(k / 32) + 1, 0u);
        w.l_.back() = std::uint32_t(1) << (k % 32);
        return w;
    }
    bool zero() const { return l_.empty(); }
    int bits() const {
        return l_.empty() ? 0 : 32 * (static_cast<int>(l_.size()) - 1) + std::bit_width(l_.back());
    }
    bool bit(int i) const { return ((l_[static_cast<std::size_t>(i / 32)] >> (i % 32)) & 1u) != 0; }
    // The value, when it fits 64 bits (callers compare first).
    std::uint64_t u64() const {
        std::uint64_t v = 0;
        for (std::size_t i = l_.size(); i-- > 0;) v = (v << 32) | l_[i];
        return v;
    }
    friend std::strong_ordering operator<=>(const Wide& a, const Wide& b) {
        if (a.l_.size() != b.l_.size()) return a.l_.size() <=> b.l_.size();
        for (std::size_t i = a.l_.size(); i-- > 0;)
            if (a.l_[i] != b.l_[i]) return a.l_[i] <=> b.l_[i];
        return std::strong_ordering::equal;
    }
    friend bool operator==(const Wide& a, const Wide& b) { return a.l_ == b.l_; }
    // *this -= b, for b <= *this.
    void sub(const Wide& b) {
        std::uint64_t borrow = 0;
        for (std::size_t i = 0; i < l_.size(); ++i) {
            const std::uint64_t s = (i < b.l_.size() ? b.l_[i] : 0u) + borrow;
            borrow = l_[i] < s ? 1u : 0u;
            l_[i] = static_cast<std::uint32_t>((std::uint64_t(l_[i]) + (borrow << 32)) - s);
        }
        trim();
    }
    // *this = 2 * *this + low_bit.
    void shift_in(bool low_bit) {
        std::uint32_t carry = low_bit ? 1u : 0u;
        for (std::uint32_t& limb : l_) {
            const std::uint32_t out = limb >> 31;
            limb = (limb << 1) | carry;
            carry = out;
        }
        if (carry != 0) l_.push_back(carry);
    }
    Wide times(std::uint64_t m) const {
        Wide w;
        w.l_.assign(l_.size() + 2, 0u);
        const std::uint64_t half[2] = {m & 0xffffffffu, m >> 32};
        for (std::size_t j = 0; j < 2; ++j) {
            std::uint64_t carry = 0;
            for (std::size_t i = 0; i < l_.size(); ++i) {
                const std::uint64_t t = std::uint64_t(l_[i]) * half[j] + w.l_[i + j] + carry;
                w.l_[i + j] = static_cast<std::uint32_t>(t);
                carry = t >> 32;
            }
            for (std::size_t k = l_.size() + j; carry != 0; ++k) {
                const std::uint64_t t = std::uint64_t(w.l_[k]) + carry;
                w.l_[k] = static_cast<std::uint32_t>(t);
                carry = t >> 32;
            }
        }
        w.trim();
        return w;
    }
    // Python's a // b and a % b (b > 0), by binary long division.
    static void divmod(const Wide& a, const Wide& b, Wide& q, Wide& r) {
        q = Wide();
        r = Wide();
        const int nbits = a.bits();
        q.l_.assign(a.l_.size(), 0u);
        for (int i = nbits - 1; i >= 0; --i) {
            r.shift_in(a.bit(i));
            if (r >= b) {
                r.sub(b);
                q.l_[static_cast<std::size_t>(i / 32)] |= std::uint32_t(1) << (i % 32);
            }
        }
        q.trim();
    }

private:
    void trim() {
        while (!l_.empty() && l_.back() == 0) l_.pop_back();
    }
    std::vector<std::uint32_t> l_;
};

}  // namespace

Rational::Rational(std::int64_t num, std::int64_t den) {
    if (den == 0) throw ZeroDivision("Fraction(" + std::to_string(num) + ", 0)");
    check_range(num);
    check_range(den);
    const std::int64_t g = gcd(num, den);
    num /= g;
    den /= g;
    if (den < 0) {
        num = -num;
        den = -den;
    }
    num_ = num;
    den_ = den;
}

Rational Rational::from_double(double x) {
    if (std::isnan(x)) throw std::invalid_argument("cannot convert NaN to integer ratio");
    if (std::isinf(x)) throw std::overflow_error("cannot convert Infinity to integer ratio");
    const std::uint64_t bits = std::bit_cast<std::uint64_t>(x);
    const bool neg = (bits >> 63) != 0;
    const int e = static_cast<int>((bits >> 52) & 0x7ffu);
    std::uint64_t sig = bits & 0x000fffffffffffffULL;
    int e2;   // |x| = sig * 2^e2
    if (e == 0) {
        e2 = -1074;
    } else {
        sig |= std::uint64_t(1) << 52;
        e2 = e - 1075;
    }
    if (sig == 0) return Rational();
    const int tz = std::countr_zero(sig);
    sig >>= tz;
    e2 += tz;
    std::int64_t num, den = 1;
    if (e2 >= 0) {
        if (std::bit_width(sig) + e2 > 63) throw std::overflow_error("Fraction(x) of this float does not fit int64");
        num = static_cast<std::int64_t>(sig << e2);
    } else {
        if (-e2 > 62) throw std::overflow_error("Fraction(x) of this float does not fit int64");
        num = static_cast<std::int64_t>(sig);
        den = std::int64_t(1) << -e2;
    }
    // sig is odd (or e2 == 0), so num/den is already in lowest terms.
    return Rational(neg ? -num : num, den, Coprime{});
}

Rational Rational::parse(std::string_view text) {
    const auto invalid = [&]() -> Rational {
        throw std::invalid_argument("Invalid literal for Fraction: " + py_repr(text));
    };
    const std::size_t n = text.size();
    std::size_t i = 0;
    skip_space(text, i);
    bool neg = false;
    if (i < n && (text[i] == '+' || text[i] == '-')) neg = text[i++] == '-';
    // (?=\d|\.\d)
    if (!(i < n && (is_digit(text[i]) || (text[i] == '.' && i + 1 < n && is_digit(text[i + 1])))))
        return invalid();
    std::string num_digits, den_digits, dec_digits, exp_digits;
    digit_groups(text, i, num_digits);   // possibly empty, before ".d"
    bool has_den = false, exp_neg = false, has_exp = false;
    // Branch 1: \s*/\s*denom
    std::size_t j = i;
    skip_space(text, j);
    if (j < n && text[j] == '/') {
        ++j;
        skip_space(text, j);
        if (!digit_groups(text, j, den_digits)) return invalid();
        has_den = true;
        i = j;
    } else {
        // Branch 2: (\.decimal)? (E[-+]?exp)?
        if (i < n && text[i] == '.') {
            ++i;
            digit_groups(text, i, dec_digits);
        }
        if (i < n && (text[i] == 'e' || text[i] == 'E')) {
            std::size_t k = i + 1;
            if (k < n && (text[k] == '+' || text[k] == '-')) exp_neg = text[k++] == '-';
            if (!digit_groups(text, k, exp_digits)) return invalid();
            has_exp = true;
            i = k;
        }
    }
    skip_space(text, i);
    if (i != n) return invalid();

    if (has_den) {
        const std::int64_t num = to_int64(num_digits.empty() ? "0" : num_digits);
        const std::int64_t den = to_int64(den_digits);
        return Rational(neg ? -num : num, den);   // ZeroDivision "Fraction(num, 0)" for /0
    }
    // value = digits * 10^p10 with digits = int part + fraction digits.
    std::string digits = num_digits + dec_digits;
    const std::size_t first = digits.find_first_not_of('0');
    if (first == std::string::npos) return Rational();   // zero, whatever the exponent
    digits.erase(0, first);
    // The exponent saturates: any |exp| beyond 10^12 overflows a nonzero value.
    std::int64_t exp = 0;
    if (has_exp) {
        for (const char c : exp_digits) {
            exp = exp * 10 + (c - '0');
            if (exp > 1000000000000LL) overflow("Fraction(str)");
        }
        if (exp_neg) exp = -exp;
    }
    std::int64_t p10 = exp - static_cast<std::int64_t>(dec_digits.size());
    while (digits.back() == '0') {
        digits.pop_back();
        ++p10;
    }
    std::int64_t num = to_int64(digits);
    std::int64_t den = 1;
    if (p10 > 0) {
        if (p10 > 18) overflow("Fraction(str)");   // num >= 1, and 10^19 > 2^63
        num = mul(num, pow_checked(10, p10), "Fraction(str)");
    } else if (p10 < 0) {
        // den = 2^k 5^k; take out the factors of 2 and 5 num shares first,
        // so a value whose reduced form fits is not refused.
        std::int64_t k2 = -p10, k5 = -p10;
        while (k2 > 0 && num % 2 == 0) {
            num /= 2;
            --k2;
        }
        while (k5 > 0 && num % 5 == 0) {
            num /= 5;
            --k5;
        }
        if (k2 > 62 || k5 > 27) overflow("Fraction(str)");
        den = mul(pow_checked(2, k2), pow_checked(5, k5), "Fraction(str)");
    }
    return Rational(neg ? -num : num, den);
}

Rational Rational::limit_denominator(std::int64_t max_den) const {
    if (max_den < 1) throw std::invalid_argument("max_denominator should be at least 1");
    if (den_ <= max_den) return *this;
    std::int64_t p0 = 0, q0 = 1, p1 = 1, q1 = 0;
    std::int64_t n = num_, d = den_;
    for (;;) {
        const std::int64_t a = floordiv(n, d);
        // q2 = q0 + a*q1 > max_den? On the first pass q1 == 0 and q2 = 1;
        // after it n, d > 0 so a >= 0, and q0 <= max_den.
        if (q1 != 0 && a > (max_den - q0) / q1) break;
        const std::int64_t q2 = q0 + a * q1;
        const std::int64_t p2 = add(p0, mul(a, p1, "limit_denominator"), "limit_denominator");
        p0 = p1;
        q0 = q1;
        p1 = p2;
        q1 = q2;
        const std::int64_t r = floormod(n, d);
        n = d;
        d = r;
        if (d == 0) break;   // exact; unreachable while den_ > max_den
    }
    const std::int64_t k = (max_den - q0) / q1;
    const std::int64_t qb = q0 + k * q1;   // <= max_den
    // CPython: 2*d*(q0+k*q1) <= self._denominator, without the product.
    if (d <= (den_ / 2) / qb) return Rational(p1, q1, Coprime{});
    return Rational(add(p0, mul(k, p1, "limit_denominator"), "limit_denominator"), qb, Coprime{});
}

Rational Rational::from_double_limited(double x, std::int64_t max_den) {
    if (std::isnan(x)) throw std::invalid_argument("cannot convert NaN to integer ratio");
    if (std::isinf(x)) throw std::overflow_error("cannot convert Infinity to integer ratio");
    if (max_den < 1) throw std::invalid_argument("max_denominator should be at least 1");
    const std::uint64_t bits = std::bit_cast<std::uint64_t>(x);
    const bool neg = (bits >> 63) != 0;
    const int e = static_cast<int>((bits >> 52) & 0x7ffu);
    std::uint64_t sig = bits & 0x000fffffffffffffULL;
    int e2 = e == 0 ? -1074 : e - 1075;
    if (e != 0) sig |= std::uint64_t(1) << 52;
    if (sig == 0) return Rational();
    const int tz = std::countr_zero(sig);
    sig >>= tz;
    e2 += tz;
    // The exact ratio fits int64 (or is an integer too big for any answer).
    if (e2 >= -62) return from_double(x).limit_denominator(max_den);

    // x = +-sig / 2^K, sig odd and < 2^53, K >= 63 > log2(max_den). CPython:
    //   p0, q0, p1, q1 = 0, 1, 1, 0; n, d = num, den
    //   loop: a = n//d; q2 = q0+a*q1; break if q2 > max_den; ...; n, d = d, n-a*d
    // The first pass has q1 == 0 (q2 = 1) and a = floor(+-sig/2^K) = 0 or -1.
    const Wide den = Wide::pow2(-e2);
    const std::int64_t a0 = neg ? -1 : 0;
    std::int64_t p0 = 1, q0 = 0, p1 = a0, q1 = 1;
    Wide n = den;
    Wide d(sig);
    if (neg) {   // num - (-1)*den = den - sig
        d = den;
        d.sub(Wide(sig));
    }
    for (;;) {
        Wide a, r;
        Wide::divmod(n, d, a, r);
        const std::int64_t lim = (max_den - q0) / q1;   // q2 > max_den  <=>  a > lim
        if (a > Wide(static_cast<std::uint64_t>(lim))) break;
        const auto ai = static_cast<std::int64_t>(a.u64());
        const std::int64_t q2 = q0 + ai * q1;
        const std::int64_t p2 = add(p0, mul(ai, p1, "limit_denominator"), "limit_denominator");
        p0 = p1;
        q0 = q1;
        p1 = p2;
        q1 = q2;
        n = d;
        d = r;
        if (d.zero()) break;   // exact; unreachable while den > max_den
    }
    const std::int64_t k = (max_den - q0) / q1;
    const std::int64_t qb = q0 + k * q1;
    // 2*d*(q0+k*q1) <= self._denominator
    if (d.times(2u * static_cast<std::uint64_t>(qb)) <= den) return Rational(p1, q1, Coprime{});
    return Rational(add(p0, mul(k, p1, "limit_denominator"), "limit_denominator"), qb, Coprime{});
}

std::string Rational::str() const {
    if (den_ == 1) return std::to_string(num_);
    return std::to_string(num_) + "/" + std::to_string(den_);
}

double Rational::to_double() const {
    if (num_ == 0) return 0.0;
    const std::uint64_t a = uabs(num_), b = static_cast<std::uint64_t>(den_);
    // a/b = (q + r/b) * 2^-e; take q to at least 55 bits, then round to 53.
    std::uint64_t q = a / b, r = a % b;
    int e = 0;
    while (q < (std::uint64_t(1) << 54)) {
        const std::uint64_t r2 = r << 1;   // r < b < 2^63
        q <<= 1;
        if (r2 >= b) {
            q |= 1u;
            r = r2 - b;
        } else {
            r = r2;
        }
        ++e;
    }
    const int s = std::bit_width(q) - 53;   // >= 2
    const std::uint64_t low = q & ((std::uint64_t(1) << s) - 1);
    const std::uint64_t halfway = std::uint64_t(1) << (s - 1);
    q >>= s;
    e -= s;
    if (low > halfway || (low == halfway && (r != 0 || (q & 1u)))) ++q;
    const double v = std::ldexp(static_cast<double>(q), -e);   // q <= 2^53: exact
    return num_ < 0 ? -v : v;
}

std::int64_t Rational::trunc() const { return num_ / den_; }
std::int64_t Rational::floor() const { return floordiv(num_, den_); }
std::int64_t Rational::ceil() const { return -floordiv(-num_, den_); }

std::int64_t Rational::round() const {
    const std::int64_t fl = floordiv(num_, den_);
    const std::int64_t rem = floormod(num_, den_);
    const std::int64_t other = den_ - rem;   // 2*rem vs den, without the product
    if (rem < other) return fl;
    if (rem > other) return fl + 1;
    return (fl % 2 == 0) ? fl : fl + 1;
}

Rational Rational::operator-() const { return Rational(-num_, den_, Coprime{}); }

// CPython's _add and _sub: the gcd of the denominators keeps the products small.
static Rational add_sub(const Rational& a, const Rational& b, bool minus) {
    const std::int64_t na = a.num(), da = a.den();
    const std::int64_t nb = minus ? -b.num() : b.num(), db = b.den();
    const char* where = minus ? "Fraction subtraction" : "Fraction addition";
    const std::int64_t g = gcd(da, db);
    if (g == 1)
        return Rational(add(mul(na, db, where), mul(da, nb, where), where), mul(da, db, where));
    const std::int64_t s = da / g;
    const std::int64_t t = add(mul(na, db / g, where), mul(nb, s, where), where);
    const std::int64_t g2 = gcd(t, g);
    if (g2 == 1) return Rational(t, mul(s, db, where));
    return Rational(t / g2, mul(s, db / g2, where));
}

Rational operator+(const Rational& a, const Rational& b) { return add_sub(a, b, false); }
Rational operator-(const Rational& a, const Rational& b) { return add_sub(a, b, true); }

Rational operator*(const Rational& a, const Rational& b) {
    std::int64_t na = a.num_, da = a.den_, nb = b.num_, db = b.den_;
    const std::int64_t g1 = gcd(na, db);
    if (g1 > 1) {
        na /= g1;
        db /= g1;
    }
    const std::int64_t g2 = gcd(nb, da);
    if (g2 > 1) {
        nb /= g2;
        da /= g2;
    }
    return Rational(mul(na, nb, "Fraction multiplication"), mul(db, da, "Fraction multiplication"),
                    Rational::Coprime{});
}

Rational operator/(const Rational& a, const Rational& b) {
    std::int64_t nb = b.num_, db = b.den_;
    if (nb == 0) throw ZeroDivision("Fraction(" + std::to_string(db) + ", 0)");
    std::int64_t na = a.num_, da = a.den_;
    const std::int64_t g1 = gcd(na, nb);
    if (g1 > 1) {
        na /= g1;
        nb /= g1;
    }
    const std::int64_t g2 = gcd(db, da);
    if (g2 > 1) {
        da /= g2;
        db /= g2;
    }
    std::int64_t n = mul(na, db, "Fraction division"), d = mul(nb, da, "Fraction division");
    if (d < 0) {
        n = -n;
        d = -d;
    }
    return Rational(n, d, Rational::Coprime{});
}

std::strong_ordering operator<=>(const Rational& x, const Rational& y) {
    // a/b against c/d (b, d > 0): integer parts first, then the reciprocals
    // of the fractional parts, the other way round. Euclid on both, so it
    // ends, and nothing is multiplied.
    std::int64_t a = x.num_, b = x.den_, c = y.num_, d = y.den_;
    for (;;) {
        const std::int64_t qa = floordiv(a, b), qc = floordiv(c, d);
        if (qa != qc) return qa <=> qc;
        const std::int64_t ra = floormod(a, b), rc = floormod(c, d);
        if (ra == 0 || rc == 0) return ra <=> rc;   // 0 < anything positive
        // ra/b vs rc/d  ==  d/rc vs b/ra
        const std::int64_t nb = b;
        a = d;
        b = rc;
        c = nb;
        d = ra;
    }
}

}  // namespace sstvae::qrss
