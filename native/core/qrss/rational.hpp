// The part of Python's fractions.Fraction that sstvae/qrss uses, on int64:
// carriers as exact rationals (ce.carrier_fraction:
// Fraction(float).limit_denominator(1 << 20)), the channeliser's mix
// frequency on its 1/64 Hz grid, and PassResult's "num/den" text.
//
// Every result is what CPython's Fraction gives, to the bit: the value is
// always normalised (gcd 1, denominator > 0), limit_denominator is
// CPython's algorithm step for step (including its tie rule), float() is
// correctly rounded like Python's int / int, and the text forms are
// str(Fraction) and Fraction(str).
//
// The one difference is range: Python's integers are unbounded and these
// are int64. Anything that would leave int64 -- an operand, an
// intermediate product, a parsed numerator before reduction -- throws
// std::overflow_error instead of answering. The numerator is kept in
// [-(2^63 - 1), 2^63 - 1] so negation never overflows. Within that, no
// answer differs: in the 300-2700 Hz carrier band every double's exact
// fraction has a denominator of at most 2^44, and from_double_limited
// covers the doubles whose exact fraction is wider than int64 but whose
// limited one is not (a tone of a few millihertz, say).
//
// Errors carry Python's text and map to Python's types in the bindings:
// ZeroDivision -> ZeroDivisionError, std::invalid_argument -> ValueError,
// std::overflow_error -> OverflowError.

#ifndef SSTVAE_QRSS_RATIONAL_HPP
#define SSTVAE_QRSS_RATIONAL_HPP

#include <compare>
#include <concepts>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

namespace sstvae::qrss {

// Python's ZeroDivisionError ("Fraction(1, 0)").
struct ZeroDivision : std::domain_error {
    using std::domain_error::domain_error;
};

class Rational {
public:
    constexpr Rational() = default;
    // Fraction(num, den), normalised. ZeroDivision for den == 0.
    Rational(std::int64_t num, std::int64_t den);
    // Fraction(n) for an integer.
    template <std::integral I>
    Rational(I n) : Rational(static_cast<std::int64_t>(checked_integer(n)), 1) {}  // NOLINT: implicit like Python ints

    // Fraction(x) for a float: the exact value (float.as_integer_ratio).
    // NaN: std::invalid_argument "cannot convert NaN to integer ratio";
    // inf: std::overflow_error "cannot convert Infinity to integer ratio";
    // a value whose exact ratio leaves int64: std::overflow_error.
    static Rational from_double(double x);

    // Fraction(text): "[sign]num[/den]" or a decimal "[sign]int[.frac][e[sign]exp]",
    // with surrounding whitespace and Python's digit-group underscores.
    // std::invalid_argument "Invalid literal for Fraction: '<text>'" otherwise.
    // `text` is UTF-8. Whitespace is Python's (str.isspace, Unicode
    // included); digits are ASCII only, where Python also takes other
    // Unicode decimal digits. A value that leaves int64 -- even one that
    // only an unreduced intermediate would -- is std::overflow_error.
    static Rational parse(std::string_view text);

    std::int64_t num() const { return num_; }
    std::int64_t den() const { return den_; }

    // Fraction.limit_denominator: the closest fraction with denominator
    // <= max_den (ties to the smaller denominator, as CPython).
    // std::invalid_argument "max_denominator should be at least 1".
    Rational limit_denominator(std::int64_t max_den = 1000000) const;

    // Fraction(x).limit_denominator(max_den) for a float -- the form every
    // sstvae/qrss caller uses -- for every finite x whose answer fits
    // int64. That includes the small ones whose exact ratio does not
    // (|x| below about 2^-10, denominator 2^63 and up), which from_double
    // refuses: CPython's loop runs on wide integers there. Errors in
    // Python's order: NaN and inf as from_double, then max_den < 1.
    static Rational from_double_limited(double x, std::int64_t max_den = 1000000);

    std::string str() const;     // str(Fraction): "num/den", or "num" when den == 1
    double to_double() const;    // float(Fraction): correctly rounded
    std::int64_t trunc() const;  // int(Fraction): toward zero
    std::int64_t floor() const;  // math.floor
    std::int64_t ceil() const;   // math.ceil
    std::int64_t round() const;  // round(Fraction): half to even

    Rational operator-() const;
    friend Rational operator+(const Rational& a, const Rational& b);
    friend Rational operator-(const Rational& a, const Rational& b);
    friend Rational operator*(const Rational& a, const Rational& b);
    friend Rational operator/(const Rational& a, const Rational& b);   // ZeroDivision "Fraction(den, 0)"
    Rational& operator+=(const Rational& b) { return *this = *this + b; }
    Rational& operator-=(const Rational& b) { return *this = *this - b; }
    Rational& operator*=(const Rational& b) { return *this = *this * b; }
    Rational& operator/=(const Rational& b) { return *this = *this / b; }

    friend bool operator==(const Rational& a, const Rational& b) {
        return a.num_ == b.num_ && a.den_ == b.den_;
    }
    // Exact, with no product that can overflow (a continued-fraction walk).
    friend std::strong_ordering operator<=>(const Rational& a, const Rational& b);

private:
    template <std::integral I>
    static std::int64_t checked_integer(I n) {
        if constexpr (std::is_signed_v<I>) {
            if (static_cast<long long>(n) < -std::numeric_limits<std::int64_t>::max())
                throw std::overflow_error("Rational: integer out of range");
        } else {
            if (static_cast<unsigned long long>(n) >
                static_cast<unsigned long long>(std::numeric_limits<std::int64_t>::max()))
                throw std::overflow_error("Rational: integer out of range");
        }
        return static_cast<std::int64_t>(n);
    }
    struct Coprime {};
    constexpr Rational(std::int64_t num, std::int64_t den, Coprime) : num_(num), den_(den) {}

    std::int64_t num_ = 0;
    std::int64_t den_ = 1;
};

}  // namespace sstvae::qrss

#endif  // SSTVAE_QRSS_RATIONAL_HPP
