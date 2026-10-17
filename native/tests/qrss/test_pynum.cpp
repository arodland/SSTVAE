// native/core/qrss/pynum against Python's int() and float() on text, the
// rules argparse's type=int and qrss_encode.py's int(text, 16) apply.
//
// Every expected value below is what CPython 3.13 returns:
//   str(int(text, base)), or None where it raises ValueError
//   float(text).hex(), or ValueError
// tests/test_qrss_native_cli.py compares the two directly, over more text.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <optional>
#include <stdexcept>
#include <string>

#include "check.hpp"
#include "qrss/pynum.hpp"

using namespace sstvae;
namespace pn = sstvae::qrss::pynum;

namespace {

struct IntCase {
    const char* text;
    int base;
    const char* decimal;   // nullptr: int() raises ValueError
};

// The C strings split after each \x escape ("\x0b""7") so the next
// character is not read as more hex digits.
constexpr IntCase INTS[] = {
    {"", 10, nullptr},
    {" ", 10, nullptr},
    {"0", 10, "0"},
    {"-0", 10, "0"},
    {"+5", 10, "5"},
    {"  12  ", 10, "12"},
    {"\x09""\x0a""7\x0b""\x0c""\x0d""", 10, "7"},
    {"\x1c""5", 10, nullptr},   // str.isspace(), but not stripped by int()
    {"1_000", 10, "1000"},
    {"1__0", 10, nullptr},
    {"_1", 10, nullptr},
    {"1_", 10, nullptr},
    {"0x1f", 10, nullptr},
    {"0X_1F", 10, nullptr},
    {"0x", 10, nullptr},
    {"0b1", 10, nullptr},
    {"1e3", 10, nullptr},
    {"007", 10, "7"},
    {"0_0", 10, "0"},
    {"+-1", 10, nullptr},
    {"1 2", 10, nullptr},
    {"1.0", 10, nullptr},
    {"-0x10", 10, nullptr},
    {"0x__f", 10, nullptr},
    {"9223372036854775807", 10, "9223372036854775807"},
    {"9223372036854775808", 10, "9223372036854775808"},
    {"-9223372036854775808", 10, "-9223372036854775808"},
    {"-9223372036854775809", 10, "-9223372036854775809"},
    {"99999999999999999999999999999999", 10, "99999999999999999999999999999999"},
    {"0xffffffffffffffffffff", 10, nullptr},
    {"d1d8", 10, nullptr},
    {"it's", 10, nullptr},
    {"", 16, nullptr},
    {" ", 16, nullptr},
    {"0", 16, "0"},
    {"-0", 16, "0"},
    {"+5", 16, "5"},
    {"  12  ", 16, "18"},
    {"\x09""\x0a""7\x0b""\x0c""\x0d""", 16, "7"},
    {"\x1c""5", 16, nullptr},
    {"1_000", 16, "4096"},
    {"1__0", 16, nullptr},
    {"_1", 16, nullptr},
    {"1_", 16, nullptr},
    {"0x1f", 16, "31"},
    {"0X_1F", 16, "31"},
    {"0x", 16, nullptr},
    {"0b1", 16, "177"},
    {"1e3", 16, "483"},
    {"007", 16, "7"},
    {"0_0", 16, "0"},
    {"+-1", 16, nullptr},
    {"1 2", 16, nullptr},
    {"1.0", 16, nullptr},
    {"-0x10", 16, "-16"},
    {"0x__f", 16, nullptr},
    {"9223372036854775807", 16, "43132194673522839607303"},
    {"9223372036854775808", 16, "43132194673522839607304"},
    {"-9223372036854775808", 16, "-43132194673522839607304"},
    {"-9223372036854775809", 16, "-43132194673522839607305"},
    {"99999999999999999999999999999999", 16, "204169420152563078078024764459060926873"},
    {"0xffffffffffffffffffff", 16, "1208925819614629174706175"},
    {"d1d8", 16, "53720"},
    {"it's", 16, nullptr},
};

// The decimal fits int64 exactly when Python's int does.
std::optional<std::int64_t> fits(const std::string& decimal) {
    try {
        std::size_t used = 0;
        const long long v = std::stoll(decimal, &used, 10);
        if (used != decimal.size()) return std::nullopt;
        return static_cast<std::int64_t>(v);
    } catch (const std::out_of_range&) {
        return std::nullopt;
    }
}

void test_ints() {
    for (const IntCase& c : INTS) {
        const std::string what = std::string("int(") + c.text + ", " + std::to_string(c.base) + ")";
        const std::optional<pn::PyInt> got = pn::py_int(c.text, c.base);
        if (!c.decimal) {
            check::is_true(!got, what + " refused");
            continue;
        }
        check::is_true(got.has_value(), what + " accepted");
        if (!got) continue;
        check::equal(got->decimal, std::string(c.decimal), what);
        check::equal(got->value.has_value(), fits(c.decimal).has_value(), what + " fits int64");
        if (got->value && fits(c.decimal)) check::equal(*got->value, *fits(c.decimal), what + " value");
    }
    bool threw = false;
    try {
        pn::py_int("10", 8);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check::is_true(threw, "base 8 is not offered");
}

void test_refusal_texts() {
    check::equal(pn::invalid_int_literal("xyz", 16), std::string("invalid literal for int() with base 16: 'xyz'"),
                 "plain");
    check::equal(pn::invalid_int_literal("it's", 10),
                 std::string("invalid literal for int() with base 10: \"it's\""), "repr picks the quote");
    check::equal(pn::invalid_int_literal(" a\tb", 16),
                 std::string("invalid literal for int() with base 16: ' a\\tb'"), "repr escapes");
    // "%.200R": the repr (quotes included) cut to 200 characters.
    check::equal(pn::invalid_int_literal(std::string(250, 'x'), 16),
                 "invalid literal for int() with base 16: '" + std::string(199, 'x'), "repr cut to 200");
}

struct FloatCase {
    const char* text;
    double value;
    bool ok;
};

constexpr double INF = HUGE_VAL;

const FloatCase FLOATS[] = {
    {"", 0, false},
    {"0", 0x0.0p+0, true},
    {"-0", -0x0.0p+0, true},
    {"+1.5", 0x1.8p+0, true},
    {" 2.5e3 ", 0x1.388p+11, true},
    {"1_0.5e1", 0x1.a4p+6, true},
    {"1__0", 0, false},
    {"1._5", 0, false},
    {"1e_5", 0, false},
    {".5", 0x1.0p-1, true},
    {"5.", 0x1.4p+2, true},
    {".", 0, false},
    {"e5", 0, false},
    {"1e", 0, false},
    {"1e-3", 0x1.0624dd2f1a9fcp-10, true},
    {"inf", INF, true},
    {"-Infinity", -INF, true},
    {"+INF", INF, true},
    {"infinit", 0, false},
    {"0x1p3", 0, false},
    {"1e400", INF, true},
    {"1e-400", 0x0.0p+0, true},
    {"4.9e-324", 0x0.0000000000001p-1022, true},
    {"0.1", 0x1.999999999999ap-4, true},
    {"123456789012345678901234567890", 0x1.8ee90ff6c373ep+96, true},
    {"\x1c""1.5", 0, false},
    {"\x0b""1.5\x0c""", 0x1.8p+0, true},
};

void test_floats() {
    for (const FloatCase& c : FLOATS) {
        const std::string what = std::string("float(") + c.text + ")";
        const std::optional<double> got = pn::py_float(c.text);
        check::equal(got.has_value(), c.ok, what + " accepted");
        if (!got || !c.ok) continue;
        check::is_true(*got == c.value && std::signbit(*got) == std::signbit(c.value), what);
    }
    for (const char* t : {"nan", "-NaN", " NAN "}) {
        const std::optional<double> got = pn::py_float(t);
        check::is_true(got && std::isnan(*got), std::string("float(") + t + ") is nan");
    }
}

}  // namespace

int main() {
    try {
        test_ints();
        test_refusal_texts();
        test_floats();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return check::report("qrss_pynum");
}
