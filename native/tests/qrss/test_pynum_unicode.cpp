// native/core/qrss/pynum on text holding Unicode spaces and decimal digits,
// against int() and float() as Python 3.13 parses it (pins written by
// gen_pynum_unicode_pins.py; tests/test_qrss_native_pynum.py holds every
// code point against Python directly). test_pynum.cpp has the ASCII rules.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>

#include "check.hpp"
#include "pynum_unicode_pins.hpp"
#include "qrss/pynum.hpp"
#include "qrss/pyunicode_tables.hpp"

using namespace sstvae;
namespace pn = sstvae::qrss::pynum;

namespace {

void test_pins() {
    for (const auto& r : pynum_unicode_pins::rows()) {
        const std::string what = std::string("text \"") + r.text + "\"";
        const auto i10 = pn::py_int(r.text, 10);
        check::equal(i10 ? i10->decimal : std::string(), std::string(r.int10), what + " int base 10");
        const auto i16 = pn::py_int(r.text, 16);
        check::equal(i16 ? i16->decimal : std::string(), std::string(r.int16), what + " int base 16");
        const auto f = pn::py_float(r.text);
        if (*r.flt == '\0') {
            check::is_true(!f.has_value(), what + " float refused");
        } else {
            const double want = std::strtod(r.flt, nullptr);
            ++check::checks;
            if (!f || std::memcmp(&*f, &want, sizeof want) != 0)
                check::fail(what + " float", f ? std::to_string(*f) : std::string("refused"));
        }
    }
}

void test_tables() {
    // Ten consecutive digits a run, counting up from 0 (Unicode's Nd blocks).
    for (const auto& r : qrss::pyunicode::DIGITS) {
        check::equal(r.value, 0, "a digit run starts at 0");
        check::is_true((r.last - r.first + 1) % 10 == 0, "a digit run is whole decades");
    }
    check::equal(std::string(qrss::pyunicode::UNICODE_VERSION), std::string("15.1.0"), "Python 3.13's Unicode");
}

}  // namespace

int main() {
    try {
        test_pins();
        test_tables();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return check::report("qrss_core_pynum_unicode");
}
