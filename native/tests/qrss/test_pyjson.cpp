// native/core/qrss/pyjson against Python's json module: dumps byte for byte
// (pinned from json.dumps), loads of that text back to the same value, and
// loads' refusals in json's words.
//
//   json.dumps({"a": 1, "b": [1.5, nan, inf, -inf, None, True, False, 1e-7, 1e22, -0.0],
//               "s": "\u00e9\U0001f600\n\"\\\x7f\u2028", "o": {}, "l": []})

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "qrss/pyjson.hpp"
#include "qrss_check.hpp"

using namespace sstvae;
using qrss::pyjson::Value;

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

static const char* WANT =
    "{\"a\": 1, \"b\": [1.5, NaN, Infinity, -Infinity, null, true, false, 1e-07, 1e+22, -0.0], "
    "\"s\": \"\\u00e9\\ud83d\\ude00\\n\\\"\\\\\\u007f\\u2028\", \"o\": {}, \"l\": []}";

static Value sample() {
    Value v = Value::object();
    v.set("a", Value::integer(1));
    Value b = Value::array();
    for (double d : {1.5, std::nan(""), HUGE_VAL, -HUGE_VAL}) b.push_back(Value::number(d));
    b.push_back(Value::null());
    b.push_back(Value::boolean(true));
    b.push_back(Value::boolean(false));
    for (double d : {1e-7, 1e22, -0.0}) b.push_back(Value::number(d));
    v.set("b", std::move(b));
    v.set("s", Value::string("\xc3\xa9\xf0\x9f\x98\x80\n\"\\\x7f\xe2\x80\xa8"));
    v.set("o", Value::object());
    v.set("l", Value::array());
    return v;
}

static void test_dumps_loads() {
    const Value v = sample();
    check::equal(qrss::pyjson::dumps(v), std::string(WANT), "json.dumps text");
    const Value back = qrss::pyjson::loads(WANT);
    check::equal(qrss::pyjson::dumps(back), std::string(WANT), "loads then dumps");
    check::equal(back.at("s").as_string(), v.at("s").as_string(), "string with escapes and a surrogate pair");
    check::is_true(std::isnan(back.at("b").items()[1].as_double()), "NaN read back");
    check::is_true(std::signbit(back.at("b").items()[9].as_double()), "-0.0 read back");
    check::equal(back.at("a").kind() == Value::Kind::Int, true, "1 is an int");
    check::equal(back.at("b").items()[0].kind() == Value::Kind::Float, true, "1.5 is a float");
    // a repeated key keeps its last value, in its first position
    check::equal(qrss::pyjson::dumps(qrss::pyjson::loads("{\"x\": 1, \"y\": 2, \"x\": 3}")),
                 std::string("{\"x\": 3, \"y\": 2}"), "repeated key");
    // a lone surrogate survives (surrogatepass UTF-8 inside)
    check::equal(qrss::pyjson::dumps(qrss::pyjson::loads("\"\\udc80\"")), std::string("\"\\udc80\""),
                 "lone surrogate");
    check::equal(qrss::pyjson::loads(" [ 1 , 2.0e0 , \"\\/\" ] ").items()[2].as_string(), std::string("/"),
                 "whitespace and \\/");
}

static void test_refusals() {
    throws<std::invalid_argument>([] { qrss::pyjson::loads(""); }, "Expecting value: line 1 column 1 (char 0)",
                                  "empty text");
    throws<std::invalid_argument>([] { qrss::pyjson::loads("{\"a\" 1}"); },
                                  "Expecting ':' delimiter: line 1 column 6 (char 5)", "missing colon");
    throws<std::invalid_argument>([] { qrss::pyjson::loads("[1, 2] x"); }, "Extra data: line 1 column 8 (char 7)",
                                  "extra data");
    throws<std::invalid_argument>([] { qrss::pyjson::loads("\"a\nb\""); },
                                  "Invalid control character at: line 1 column 3 (char 2)", "control character");
    try {
        (void)sample().at("missing");
        check::fail("absent key", "did not throw");
    } catch (const qrss::pyjson::KeyError& e) {
        check::equal(e.key(), std::string("missing"), "KeyError's key");
        check::equal(std::string(e.what()), std::string("KeyError: 'missing'"), "KeyError's text");
    }
}

int main() {
    try {
        test_dumps_loads();
        test_refusals();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return check::report("qrss_pyjson");
}
