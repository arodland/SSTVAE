// native/core/qrss/morse (and qrss/pytext, which writes its refusals)
// against values from the Python reference.
//
// Digests are hashlib.sha256(a.tobytes()) of sstvae/qrss/morse.py's
// output; each refusal text is str(e) of the ValueError Python raises for
// the same input (UTF-8; a byte that is not UTF-8 is what os.fsdecode
// makes of it). tests/test_qrss_native_morse.py compares the two
// implementations directly where Python is available.

#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "qrss/qrss_config.hpp"
#include "qrss/morse.hpp"
#include "qrss/pytext.hpp"
#include "qrss_check.hpp"

using namespace sstvae;
namespace mo = sstvae::qrss::morse;
namespace pt = sstvae::qrss::pytext;
namespace qc = sstvae::qrss::config;
using qrss_check::digest_is;

namespace {

std::string refusal(const std::string& call) {
    try {
        mo::check_callsign(call);
    } catch (const std::invalid_argument& e) {
        return e.what();
    }
    return "(accepted)";
}

std::string bits(const std::vector<std::uint8_t>& v) {
    std::string s;
    for (std::uint8_t b : v) s += static_cast<char>('0' + b);
    return s;
}

}  // namespace

static void test_table() {
    check::equal(qc::MORSE.size(), std::size_t(37), "37 characters");
    check::equal(std::string(mo::code_of(U'A')), std::string(".-"), "A");
    check::equal(std::string(mo::code_of(U'/')), std::string("-..-."), "/");
    check::is_true(mo::code_of(U'a').empty() && mo::code_of(U' ').empty() && mo::code_of(0x0130).empty(),
                   "nothing outside [A-Z0-9/]");
}

static void test_keying() {
    // np.concatenate([morse.keying_units(c) for c in morse.MORSE]), and the same
    // of morse_keying: every character, in the table's order.
    std::vector<std::uint8_t> units, keys;
    for (const qc::MorseCode& m : qc::MORSE) {
        const std::string c(1, m.ch);
        const auto u = mo::keying_units(c);
        check::equal(u.size(), std::size_t(qc::CW_UNITS), "keying_units(" + c + ") length");
        units.insert(units.end(), u.begin(), u.end());
        const auto k = mo::morse_keying(c);
        keys.insert(keys.end(), k.begin(), k.end());
    }
    digest_is(units, "a229cc0e4465b7abd0ec05d47023004dc201b5ef70ea911dd3c6faaca32bc360", "keying_units, all 37");
    digest_is(keys, "8d1476ec31e103fc72c60e240621a53aceaf0811a97ffba212befcb27823e4f9", "morse_keying, all 37");
    digest_is(mo::keying_units("K1ABC/P"), "46c4fb4542fce939dffea24afa991d130d650e7bfca58a7e1735023750a83089",
              "keying_units(K1ABC/P)");
    check::equal(mo::morse_keying("00000000").size(), std::size_t(173), "00000000 takes 173 units");
    const auto z = mo::keying_units("00000000");
    check::is_true(z[8] == 1 && z[8 + 172] == 1 && z[8 + 173] == 0, "00000000 placement");
    check::equal(bits(mo::morse_keying("K1ABC")),
                 std::string("111010111000101110111011101110001011100011101010100011101011101"), "K1ABC");
    const auto k = mo::keying_units("K1ABC");
    bool lead_quiet = true;
    for (int u = 0; u < qc::CW_LEAD_UNITS; ++u) lead_quiet = lead_quiet && k[u] == 0;
    check::is_true(lead_quiet && k[qc::CW_LEAD_UNITS] == 1, "the call starts at unit 8");
    check::is_true(mo::keying_units("K1ABC   ") == k, "trailing spaces are stripped");
    check::equal(mo::check_callsign("K1ABC   "), std::string("K1ABC"), "check_callsign strips");
    check::equal(mo::check_callsign("W1AW/8"), std::string("W1AW/8"), "check_callsign keeps a valid call");
}

static void test_refusals() {
    const std::pair<std::string, std::string> cases[] = {
        {"k1abc", "callsign 'k1abc' has characters outside [A-Z0-9/]: ['a', 'b', 'c', 'k']"},
        {"K1ABCDEFG", "callsign 'K1ABCDEFG' must be 1 to 8 characters"},
        {"", "callsign '' must be 1 to 8 characters"},
        {"   ", "callsign '   ' must be 1 to 8 characters"},
        {"aaaaaaaaa   ", "callsign 'aaaaaaaaa   ' must be 1 to 8 characters"},
        {" K1ABC", "callsign ' K1ABC' has characters outside [A-Z0-9/]: [' ']"},
        {"K1-AB", "callsign 'K1-AB' has characters outside [A-Z0-9/]: ['-']"},
        {"K1'B", R"(callsign "K1'B" has characters outside [A-Z0-9/]: ["'"])"},
        {"K1\"B", R"(callsign 'K1"B' has characters outside [A-Z0-9/]: ['"'])"},
        {"K1'\"B", R"(callsign 'K1\'"B' has characters outside [A-Z0-9/]: ['"', "'"])"},
        {"K\\1", R"(callsign 'K\\1' has characters outside [A-Z0-9/]: ['\\'])"},
        {"K\xc3\x84\xc2\xa0", "callsign 'K\xc3\x84\\xa0' has characters outside [A-Z0-9/]: ['\\xa0', '\xc3\x84']"},
        {"K\xff\xe2\x82",
         R"(callsign 'K\udcff\udce2\udc82' has characters outside [A-Z0-9/]: ['\udc82', '\udce2', '\udcff'])"},
        {"K\t\n\x7f", R"(callsign 'K\t\n\x7f' has characters outside [A-Z0-9/]: ['\t', '\n', '\x7f'])"},
        {"\xf0\x9f\x98\x80K", "callsign '\xf0\x9f\x98\x80K' has characters outside [A-Z0-9/]: ['\xf0\x9f\x98\x80']"},
        {"\xee\x80\x80", R"(callsign '\ue000' has characters outside [A-Z0-9/]: ['\ue000'])"},
        {"A\xe2\x80\xa8", R"(callsign 'A\u2028' has characters outside [A-Z0-9/]: ['\u2028'])"},
    };
    for (const auto& [call, want] : cases) check::equal(refusal(call), want, "refusal of " + want);
    // Every entry point refuses the same way.
    bool threw = false;
    try {
        mo::keying_units("k1abc");
    } catch (const std::invalid_argument& e) {
        threw = std::string(e.what()) == cases[0].second;
    }
    check::is_true(threw, "keying_units refuses as check_callsign does");
    // The fields a caller words its own refusal from (the Python binding does).
    using Why = mo::CallsignError::Why;
    try {
        mo::morse_keying("k1a\xc3\x84" "ab");
        check::is_true(false, "lowercase is refused");
    } catch (const mo::CallsignError& e) {
        check::is_true(e.why() == Why::characters && e.bad() == std::u32string{U'a', U'b', U'k', 0xC4},
                       "CallsignError: the bad characters, sorted and unique");
    }
    try {
        mo::check_callsign("ABCDEFGHI  ");
        check::is_true(false, "nine characters are refused");
    } catch (const mo::CallsignError& e) {
        check::is_true(e.why() == Why::length && e.bad().empty(), "CallsignError: length");
    }
}

static void test_pytext() {
    check::equal(pt::decode("A\xc3\x84\xe2\x82\xac\xf0\x9f\x98\x80").size(), std::size_t(4), "decode counts code points");
    check::is_true(pt::decode("\xed\xa0\x80") == std::u32string(1, 0xD800), "surrogatepass form decodes");
    check::is_true(pt::decode("\xc0\xaf") == std::u32string{0xDCC0, 0xDCAF}, "overlong bytes escape one by one");
    check::is_true(pt::decode("\xf4\x90\x80\x80").size() == 4, "beyond U+10FFFF escapes");
    check::equal(pt::repr(std::string_view("it's")), std::string(R"("it's")"), "repr quote choice");
    check::equal(pt::repr(std::string_view("\xc2\xad\xe2\x80\x8b\xef\xbb\xbf")),
                 std::string(R"('\xad\u200b\ufeff')"), "repr escapes format characters");
    check::equal(pt::repr(std::u32string_view(U"\U000E0001")), std::string(R"('\U000e0001')"), "repr \\U form");
    check::is_true(pt::isprintable(U'~') && !pt::isprintable(U'\x7f') && !pt::isprintable(0x85) &&
                       pt::isprintable(0xA1) && !pt::isprintable(0x3000) && pt::isprintable(0x1F600),
                   "isprintable");
}

int main() {
    try {
        test_table();
        test_keying();
        test_refusals();
        test_pytext();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return check::report("qrss_morse");
}
