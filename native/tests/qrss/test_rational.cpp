// native/core/qrss/rational against Python's fractions.Fraction. Every
// expected value below is what CPython 3.13 printed for the same call;
// the digests cover 100,000 carriers in the 300-2700 Hz band:
//
//   z = splitmix(77, 100000); xs = 300.0 + unit(z) * 2400.0    # infra_testgen.hpp
//   fs = [Fraction(x).limit_denominator(1 << 20) for x in xs.tolist()]
//   sha256(np.array([v for f in fs for v in (f.numerator, f.denominator)], np.int64).tobytes())
//   sha256("\n".join(str(f) for f in fs).encode())
//   sha256(np.array([float(f) for f in fs]).tobytes())

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "infra_testgen.hpp"
#include "qrss/rational.hpp"
#include "qrss_check.hpp"
#include "util/sha256.hpp"

using namespace sstvae;
using qrss::Rational;

template <typename E, typename F>
static void throws(F f, const std::string& text, const std::string& what) {
    try {
        f();
    } catch (const E& e) {
        check::equal(std::string(e.what()), text, what);
        return;
    }
    check::fail(what, "did not throw");
}

static void is(const Rational& r, std::int64_t num, std::int64_t den, const std::string& what) {
    check::equal(r.num(), num, what + " numerator");
    check::equal(r.den(), den, what + " denominator");
}

static void test_limit_denominator() {
    struct Case {
        double x;
        std::int64_t max_den, num, den;
    };
    const Case cases[] = {
        {1500.3, 1048576, 15003, 10},
        {1234.5678, 1048576, 6172839, 5000},
        {300.1, 1048576, 3001, 10},
        {2699.9, 1048576, 26999, 10},
        {0.1, 1048576, 1, 10},
        {1500.3, 65536, 15003, 10},
        {3.141592653589793, 10, 22, 7},
        {3.141592653589793, 100, 311, 99},
        {-3.141592653589793, 100, -311, 99},
        {2.5, 1, 2, 1},     // a tie between equal denominators takes the floor
        {3.5, 1, 3, 1},
        {-2.5, 1, -3, 1},
        {0.3333333333333333, 3, 1, 3},
        {0.5, 1, 0, 1},
    };
    for (const Case& c : cases) {
        const std::string what = "limit_denominator(" + std::to_string(c.x) + ", " + std::to_string(c.max_den) + ")";
        is(Rational::from_double(c.x).limit_denominator(c.max_den), c.num, c.den, what);
        is(Rational::from_double_limited(c.x, c.max_den), c.num, c.den, what + " from the float");
    }
    // Floats whose exact ratio is wider than int64 (denominator 2^63 and
    // up); from_double refuses them, from_double_limited answers as Python.
    const Case tiny[] = {
        {0x1.b7cdfd9d7bdbbp-34, 1048576LL, 0LL, 1LL},   // 1e-10
        {-0x1.b7cdfd9d7bdbbp-34, 1048576LL, 0LL, 1LL},
        {0x1.0624dd2f1a9fcp-10, 65536LL, 1LL, 1000LL},   // 1e-3 (fits; the int64 path)
        {-0x1.0624dd2f1a9fcp-10, 65536LL, -1LL, 1000LL},
        {0x1.6f0068db8bac7p-11, 65536LL, 7LL, 10000LL},   // 7e-4
        {0x1.0000000000000p-63, 4611686018427387904LL, 0LL, 1LL},   // a tie
        {-0x1.0000000000000p-63, 4611686018427387904LL, 0LL, 1LL},
        {0x1.0000000000000p-63, 9223372036854775807LL, 1LL, 9223372036854775807LL},
        {-0x1.0000000000000p-63, 9223372036854775807LL, -1LL, 9223372036854775807LL},
        {0x0.0000000000001p-1022, 9223372036854775807LL, 0LL, 1LL},
        {-0x0.0000000000001p-1022, 1LL, 0LL, 1LL},
        {0x1.56e1fc2f8f359p-997, 1LL, 0LL, 1LL},
        {-0x1.56e1fc2f8f359p-997, 1LL, 0LL, 1LL},
        {0x1.f75104d551d69p-16, 1000000LL, 3LL, 100000LL},   // 3e-5
        {-0x1.f75104d551d69p-16, 1000000LL, -3LL, 100000LL},
        {0x1.5b7b198f1a428p-40, 9223372036854775807LL, 10232741LL, 8288976103685702439LL},
        {-0x1.5b7b198f1a428p-40, 9223372036854775807LL, -10232741LL, 8288976103685702439LL},
        {0x1.8000000000000p-69, 4611686018427387904LL, 0LL, 1LL},
        {0x1.0000000000000p-63, 7LL, 0LL, 1LL},
    };
    for (const Case& c : tiny)
        is(Rational::from_double_limited(c.x, c.max_den), c.num, c.den,
           "from_double_limited(" + std::to_string(c.x) + ", " + std::to_string(c.max_den) + ")");
    throws<std::invalid_argument>([] { Rational::from_double_limited(std::nan(""), 0); },
                                  "cannot convert NaN to integer ratio", "NaN before max_den");
    throws<std::invalid_argument>([] { Rational::from_double_limited(1e-300, 0); },
                                  "max_denominator should be at least 1", "tiny, max_den 0");
    is(Rational(4321, 8765).limit_denominator(10000), 4321, 8765, "already within the limit");
    throws<std::invalid_argument>([] { Rational(1, 3).limit_denominator(0); },
                                  "max_denominator should be at least 1", "max_den 0");
}

static void test_band_digest() {
    std::vector<std::int64_t> pairs;
    std::string text;
    std::vector<double> floats;
    for (const std::uint64_t z : qrss_testgen::splitmix(77, 100000)) {
        const Rational f = Rational::from_double(300.0 + qrss_testgen::unit(z) * 2400.0).limit_denominator(1 << 20);
        pairs.push_back(f.num());
        pairs.push_back(f.den());
        if (!text.empty()) text += '\n';
        text += f.str();
        floats.push_back(f.to_double());
        // str() parses back to itself.
        if (!(Rational::parse(f.str()) == f)) check::fail("parse(str(f))", f.str());
    }
    qrss_check::digest_is(pairs, "ed8b40a4ad791a87b8fb44320f6d5b02f1579e269776d082d29b76c231544ce4",
                          "limit_denominator over the band");
    check::equal(util::to_hex(util::sha256(std::string_view(text))),
                 std::string("eb5ca1adb1bd9ef41af0674e162c2e3649bfa2936687d91e1414d7fe232beeea"), "str over the band");
    qrss_check::digest_is(floats, "6be78552c97b5039c7c60e5642662904daf8c36fbb2ac1ed6a73beaa98b688d5",
                          "float() over the band");
}

static void test_from_double() {
    is(Rational::from_double(1500.3), 6598389180609331, 4398046511104, "1500.3 exactly");
    is(Rational::from_double(0.1), 3602879701896397, 36028797018963968, "0.1 exactly");
    is(Rational::from_double(-2.5), -5, 2, "-2.5");
    is(Rational::from_double(-0.0), 0, 1, "-0.0");
    is(Rational::from_double(1e18), 1000000000000000000, 1, "1e18");
    is(Rational::from_double(0x1p62), std::int64_t(1) << 62, 1, "2^62");
    is(Rational::from_double(0x1p-62), 1, std::int64_t(1) << 62, "2^-62");
    throws<std::invalid_argument>([] { Rational::from_double(std::nan("")); }, "cannot convert NaN to integer ratio",
                                  "NaN");
    throws<std::overflow_error>([] { Rational::from_double(-std::numeric_limits<double>::infinity()); },
                                "cannot convert Infinity to integer ratio", "inf");
    bool threw = false;
    try {
        Rational::from_double(5e-324);
    } catch (const std::overflow_error&) {
        threw = true;
    }
    check::is_true(threw, "a ratio past int64 is refused, not truncated");
}

static void test_parse() {
    struct Case {
        const char* s;
        std::int64_t num, den;
    };
    const Case cases[] = {
        {"3/2", 3, 2},
        {" -35/4 ", -35, 4},
        {"3.1415", 6283, 2000},
        {"-47e-2", -47, 100},
        {"1_000", 1000, 1},
        {"1.e5", 100000, 1},
        {".5", 1, 2},
        {"+.5e-3", 1, 2000},
        {"0.00000095367431640625", 1, 1048576},
        {"1E+0_1", 10, 1},
        {"0e-99999", 0, 1},
        {"1500.300000000000000000000000", 15003, 10},
        {"-0", 0, 1},
        {"007/0014", 1, 2},
        {"\t96017/64\n", 96017, 64},
    };
    for (const Case& c : cases) is(Rational::parse(c.s), c.num, c.den, std::string("parse ") + c.s);
    for (const char* bad : {"1._5", "1__2", "1_", "1e5/2", "1.5/2", "nan", "1 . 5", "", " ", "/2", ".", "e5", "1/",
                            "1/-2", "--1", "0x10", "1/2/3"})
        throws<std::invalid_argument>([&] { Rational::parse(bad); },
                                      std::string("Invalid literal for Fraction: '") + bad + "'",
                                      std::string("refuse ") + bad);
    throws<std::invalid_argument>([] { Rational::parse("'a'"); }, "Invalid literal for Fraction: \"'a'\"", "repr quotes");
    throws<std::invalid_argument>([] { Rational::parse("\x01\t"); }, "Invalid literal for Fraction: '\\x01\\t'",
                                  "repr escapes");
    throws<qrss::ZeroDivision>([] { Rational::parse("-5/0"); }, "Fraction(-5, 0)", "x/0");
    throws<qrss::ZeroDivision>([] { Rational::parse("-0/0"); }, "Fraction(0, 0)", "0/0");
    bool threw = false;
    try {
        Rational::parse("1e30");
    } catch (const std::overflow_error&) {
        threw = true;
    }
    check::is_true(threw, "past int64 is refused");
}

static void test_float() {
    struct Case {
        std::int64_t num, den;
        double want;
    };
    const Case cases[] = {
        {1LL, 3LL, 0x1.5555555555555p-2},
        {2LL, 3LL, 0x1.5555555555555p-1},
        {-1LL, 3LL, -0x1.5555555555555p-2},
        {15003LL, 10LL, 0x1.7713333333333p+10},
        {6172839LL, 5000LL, 0x1.34a456d5cfaadp+10},
        {4611686018427387903LL, 3LL, 0x1.5555555555555p+60},
        {1LL, 4611686018427387903LL, 0x1.0000000000000p-62},
        {9007199254740993LL, 1LL, 0x1.0000000000000p+53},   // a tie: to even
        {9007199254740995LL, 1LL, 0x1.0000000000002p+53},   // a tie: to even, upward
        {9223372036854775807LL, 4611686018427387905LL, 0x1.0000000000000p+1},
        {12207LL, 970LL, 0x1.92b48503b1e16p+3},
    };
    for (const Case& c : cases)
        check::equal(Rational(c.num, c.den).to_double(), c.want,
                     "float(" + std::to_string(c.num) + "/" + std::to_string(c.den) + ")");
    check::is_true(!std::signbit(Rational(0, 5).to_double()), "0 is +0.0");
}

static void test_rounding() {
    struct Case {
        std::int64_t num, den, trunc, floor, ceil, round;
    };
    const Case cases[] = {
        {5, 2, 2, 2, 3, 2},   {7, 2, 3, 3, 4, 4},   {-5, 2, -2, -3, -2, -2}, {-7, 2, -3, -4, -3, -4},
        {1, 3, 0, 0, 1, 0},   {-1, 3, 0, -1, 0, 0}, {2, 3, 0, 0, 1, 1},      {-2, 3, 0, -1, 0, -1},
        {9, 4, 2, 2, 3, 2},
    };
    for (const Case& c : cases) {
        const Rational r(c.num, c.den);
        const std::string w = r.str();
        check::equal(r.trunc(), c.trunc, "int(" + w + ")");
        check::equal(r.floor(), c.floor, "floor(" + w + ")");
        check::equal(r.ceil(), c.ceil, "ceil(" + w + ")");
        check::equal(r.round(), c.round, "round(" + w + ")");
    }
}

static void test_arithmetic() {
    const Rational a(96017, 64), b(-3, 10);
    check::equal((a + b).str(), std::string("479989/320"), "+");
    check::equal((a - b).str(), std::string("480181/320"), "-");
    check::equal((a * b).str(), std::string("-288051/640"), "*");
    check::equal((a / b).str(), std::string("-480085/96"), "/");
    check::equal((Rational(1, 2) - Rational(1, 2)).str(), std::string("0"), "zero is 0/1");
    is(Rational(10, -8), -5, 4, "Fraction(10, -8)");
    is(Rational(0, -5), 0, 1, "Fraction(0, -5)");
    check::equal((a * 64).str(), std::string("96017"), "times an int");
    throws<qrss::ZeroDivision>([] { Rational(1, 0); }, "Fraction(1, 0)", "Fraction(1, 0)");
    throws<qrss::ZeroDivision>([&] { (void)(a / Rational(0, 7)); }, "Fraction(1, 0)", "a / 0");
    check::is_true(Rational(1, 3) < Rational(1, 2) && Rational(-1, 2) < Rational(-1, 3) && Rational(5, 2) > 2 &&
                       Rational(4, 2) == 2 && Rational(7, 3) >= Rational(7, 3),
                   "comparisons");
    // Exact where the cross products would overflow int64.
    const std::int64_t big = std::numeric_limits<std::int64_t>::max();
    check::is_true(Rational(big - 1, big) > Rational(big - 2, big - 1), "comparison near 1 without overflow");
    check::is_true(Rational(big, big - 1) < Rational(big - 1, big - 2), "and just above 1");
    check::is_true(!(Rational(big, big - 1) > Rational(big - 1, big - 2)), "and just above 1, the other operator");
    bool threw = false;
    try {
        (void)(Rational(big, 3) * Rational(big, 5));
    } catch (const std::overflow_error&) {
        threw = true;
    }
    check::is_true(threw, "an int64 overflow throws");
}

int main() {
    try {
        test_limit_denominator();
        test_band_digest();
        test_from_double();
        test_parse();
        test_float();
        test_rounding();
        test_arithmetic();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return check::report("qrss_rational");
}
