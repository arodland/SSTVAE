// native/core/qrss/header against the Python reference (sstvae/qrss/header.py).
//
// Digests are hashlib.sha256(header.pack(h).tobytes()) and
// hashlib.sha256(header.encode(h).tobytes()) for tests/test_qrss_header.py's
// five headers; refusal texts are the reference's ValueError messages.
// tests/test_qrss_native_header.py compares the two implementations directly
// where Python is available.

#include <array>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "beacon/beacon.hpp"
#include "qrss/header.hpp"
#include "qrss/polar.hpp"
#include "qrss_check.hpp"

using namespace sstvae;
namespace hdr = sstvae::qrss::header;
using qrss_check::digest_is;

namespace {

hdr::HeaderFields make(std::string call, std::optional<std::string> grid, std::int64_t pid, std::int64_t mode,
                       std::int64_t seg, std::int64_t codec, std::int64_t version = 2, std::int64_t reserved = 0) {
    hdr::HeaderFields h;
    h.callsign = std::move(call);
    h.grid = std::move(grid);
    h.picture_id = pid;
    h.mode = mode;
    h.segment = seg;
    h.codec_id = codec;
    h.version = version;
    h.reserved = reserved;
    return h;
}

template <typename F>
std::string refusal(F&& f) {
    try {
        f();
    } catch (const std::invalid_argument& e) {
        return e.what();
    }
    return "(no refusal)";
}

std::uint64_t to_int(std::span<const std::uint8_t> bits) {
    std::uint64_t v = 0;
    for (std::uint8_t b : bits) v = (v << 1) | b;
    return v;
}

std::vector<std::uint8_t> with_crc(std::vector<std::uint8_t> body) {
    const auto crc = hdr::crc16(body);
    body.insert(body.end(), crc.begin(), crc.end());
    return body;
}

void set_field(std::vector<std::uint8_t>& bits, int a, int b, std::uint64_t v) {
    for (int i = b - 1; i >= a; --i, v >>= 1) bits[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(v & 1);
}

const hdr::HeaderFields H = make("K1ABC/P", "FN42", 0xDEADBEEF, 2, 1, 0xD1D8);

}  // namespace

static void test_crc() {
    // The beacon CRC over ASCII "123456789" is 0xA69D (CCITT-FALSE would be 0x29B1).
    std::vector<std::uint8_t> bits;
    for (char c : std::string("123456789"))
        for (int k = 7; k >= 0; --k) bits.push_back(static_cast<std::uint8_t>((static_cast<unsigned>(c) >> k) & 1));
    check::equal(to_int(hdr::crc16(bits)), std::uint64_t(0xA69D), "crc16 check value");
    check::equal(to_int(hdr::crc16({})), std::uint64_t(0xFFFF), "crc16 of nothing");
    check::equal(refusal([] { hdr::crc16(std::vector<std::uint8_t>{0, 2}); }),
                 std::string("crc16 takes bits of 0 or 1"), "crc16 refuses a 2");
}

static void test_layout() {
    const std::pair<const char*, std::pair<int, int>> want[] = {
        {"callsign", {0, 48}}, {"grid", {48, 63}}, {"picture_id", {63, 95}}, {"mode", {95, 97}},
        {"segment", {97, 99}}, {"codec_id", {99, 115}}, {"version", {115, 119}}, {"reserved", {119, 126}},
        {"crc", {126, 142}}};
    for (const auto& [name, span] : want) check::is_true(hdr::offset(name) == span, std::string("offset ") + name);
    const auto bits = hdr::pack(H);
    const std::span<const std::uint8_t> b(bits);
    std::vector<int> codes;
    for (int i = 0; i < 8; ++i) codes.push_back(static_cast<int>(to_int(b.subspan(static_cast<std::size_t>(6 * i), 6))));
    check::equal(beacon::codes_to_callsign(codes), std::string("K1ABC/P"), "callsign field");
    check::equal(to_int(b.subspan(48, 15)), std::uint64_t(((5 * 18 + 13) * 10 + 4) * 10 + 2), "grid field");
    check::equal(to_int(b.subspan(63, 32)), std::uint64_t(0xDEADBEEF), "picture field");
    check::equal(to_int(b.subspan(95, 2)), std::uint64_t(2), "mode field");
    check::equal(to_int(b.subspan(97, 2)), std::uint64_t(1), "segment field");
    check::equal(to_int(b.subspan(99, 16)), std::uint64_t(0xD1D8), "codec field");
    check::equal(to_int(b.subspan(115, 4)), std::uint64_t(2), "version field");
    check::equal(to_int(b.subspan(119, 7)), std::uint64_t(0), "reserved field");
    check::is_true(hdr::crc16(b.first(126)) == std::vector<std::uint8_t>(bits.begin() + 126, bits.end()), "crc field");
}

static void test_pack_digests() {
    struct Case {
        hdr::HeaderFields h;
        const char* pack;
        const char* encode;
    };
    const Case cases[] = {
        {H, "d3a2d7aa952346d891536d0d223d86b0d7eba1b2e5e8f66bd1c623ea980d075a",
         "f894a2ec30bbb14e43ec7cfa730e64c3add67897efb6c7226c332deaed381651"},
        {make("K1ABC", std::nullopt, 0, 0, 0, 0), "a72593c60c4454904a10cd256f18a7ab6d0d02f532ca159283d36947285aed70",
         "ea9456e8c112407785a200845b35470a74bdd5097d707f7ee362422e8f29c333"},
        {make("G", "AA00", 1, 1, 0, 1), "5e5e6c0e582785f8bccebf5c3782b65a824d443223b753e3c7b5bbddb9699af5",
         "9fe06b266b6eee3e8c8a84d08a849749226b3ead8d7a0c360c14f84a07661460"},
        {make("VK2/W1AW", "RR99", 0xFFFFFFFF, 2, 2, 0xFFFF),
         "8e43efc38ee338afffa87378ffa5e97046d39a5b7812809c7df6c953ac67adf2",
         "9c81b7a345453f75709c05360020ceb622865dbea7b4640409bd7be785f5c23e"},
        {make("00000000", "JO62", 0x12345678, 1, 1, 0xD1D8),
         "ce40c8b73052cada4bd50afd64d2ce25975ab990a37bb8add499681987481099",
         "91756e02de5424d23bc8a072262cbaaaf81b219791e871ec4e6bbd0c37fe0807"},
    };
    for (const auto& c : cases) {
        const auto bits = hdr::pack(c.h);
        digest_is(bits, c.pack, "pack " + c.h.callsign);
        digest_is(hdr::encode(c.h), c.encode, "encode " + c.h.callsign);
        check::is_true(hdr::unpack(bits) == std::optional<hdr::HeaderFields>(c.h), "unpack round trip " + c.h.callsign);
        // Noiseless decode, list 8 and plain SC.
        const auto coded = hdr::encode(c.h);
        std::vector<double> llr(coded.size());
        for (std::size_t e = 0; e < coded.size(); ++e) llr[e] = 10.0 * (1.0 - 2.0 * coded[e]);
        check::is_true(hdr::decode(llr) == std::optional<hdr::HeaderFields>(c.h), "decode " + c.h.callsign);
        check::is_true(hdr::decode(llr, 1) == std::optional<hdr::HeaderFields>(c.h), "decode L=1 " + c.h.callsign);
    }
}

static void test_unpack_rejects() {
    const auto bits = hdr::pack(H);
    bool all = true;
    for (std::size_t i = 0; i < bits.size(); ++i) {
        auto f = bits;
        f[i] ^= 1;
        all = all && !hdr::unpack(f);
    }
    check::is_true(all, "every single-bit flip is rejected");
    // CRC-valid bodies with one bad field.
    const struct {
        int a, b;
        std::uint64_t v;
        const char* what;
    } bad[] = {{115, 119, 1, "version 1 (only on request)"}, {115, 119, 3, "version 3"}, {119, 126, 1, "reserved"},
               {95, 97, 3, "mode 3"}, {97, 99, 3, "segment above mode"}, {48, 63, 32400, "unused grid code"}};
    for (const auto& x : bad) {
        std::vector<std::uint8_t> body(bits.begin(), bits.begin() + 126);
        set_field(body, x.a, x.b, x.v);
        check::is_true(!hdr::unpack(with_crc(body)), std::string("CRC-valid ") + x.what + " is rejected");
    }
    // Version 1 is accepted when asked for.
    std::vector<std::uint8_t> body(bits.begin(), bits.begin() + 126);
    set_field(body, 115, 119, 1);
    const std::array<int, 2> both{1, 2};
    auto v1 = H;
    v1.version = 1;
    check::is_true(hdr::unpack(with_crc(body), both) == std::optional<hdr::HeaderFields>(v1), "version 1 on request");
    // Callsigns the alphabet can carry and QRSS refuses: '-', '?', an internal space, empty.
    for (const char* call : {"K1-AB", "K1 AB", "", "        ", "K1AB?"}) {
        std::vector<std::uint8_t> b2(bits.begin(), bits.begin() + 126);
        const auto codes = beacon::callsign_to_codes(call);
        for (int i = 0; i < 8; ++i) set_field(b2, 6 * i, 6 * i + 6, static_cast<std::uint64_t>(codes[static_cast<std::size_t>(i)]));
        check::is_true(!hdr::unpack(with_crc(b2)), std::string("callsign '") + call + "' is rejected");
    }
    check::is_true(!hdr::unpack(std::vector<std::uint8_t>(141, 0)), "141 bits rejected");
    check::is_true(!hdr::unpack(std::vector<std::uint8_t>(142, 2)), "a value of 2 rejected");
}

static void test_grid() {
    bool ok = true;
    for (int v = 0; v < hdr::N_GRID_CODES && ok; ++v) {
        const auto g = hdr::grid_decode(v);
        ok = g && g->size() == 4 && hdr::grid_encode(*g) == v;
    }
    check::is_true(ok, "every locator round trips");
    check::is_true(!hdr::grid_decode(32767) && hdr::grid_encode(std::nullopt) == 32767, "none");
    check::equal(hdr::grid_encode(std::string_view("fn42")), hdr::grid_encode(std::string_view("FN42")), "case");
    check::equal(*hdr::grid_decode(((5 * 18 + 13) * 10 + 4) * 10 + 2), std::string("FN42"), "FN42");
    for (const char* bad : {"FN4", "FN421", "SN42", "F142", "AAAA"})
        check::equal(refusal([&] { hdr::grid_encode(std::string_view(bad)); }),
                     std::string("grid must be a 4-character Maidenhead locator, got '") + bad + "'",
                     std::string("grid refusal ") + bad);
    for (std::int64_t bad : {std::int64_t(32400), std::int64_t(32766), std::int64_t(-1), std::int64_t(32768)})
        check::equal(refusal([&] { hdr::grid_decode(bad); }), "invalid grid code " + std::to_string(bad),
                     "grid code refusal " + std::to_string(bad));
}

static void test_pack_refusals() {
    const struct {
        hdr::HeaderFields h;
        const char* text;
    } cases[] = {
        {make("k1abc", std::nullopt, 0, 0, 0, 0), "callsign must be 1-8 characters of [A-Z0-9/], got 'k1abc'"},
        {make("K1ABCDEFG", std::nullopt, 0, 0, 0, 0),
         "callsign must be 1-8 characters of [A-Z0-9/], got 'K1ABCDEFG'"},
        {make("K1 AB", std::nullopt, 0, 0, 0, 0), "callsign must be 1-8 characters of [A-Z0-9/], got 'K1 AB'"},
        {make("", std::nullopt, 0, 0, 0, 0), "callsign must be 1-8 characters of [A-Z0-9/], got ''"},
        {make("K1'B", std::nullopt, 0, 0, 0, 0), "callsign must be 1-8 characters of [A-Z0-9/], got \"K1'B\""},
        {make("K1ABC", "XX99", 0, 0, 0, 0), "grid must be a 4-character Maidenhead locator, got 'XX99'"},
        {make("K1ABC", std::nullopt, std::int64_t(1) << 32, 0, 0, 0), "picture_id out of range: 4294967296"},
        {make("K1ABC", std::nullopt, -1, 0, 0, 0), "picture_id out of range: -1"},
        {make("K1ABC", std::nullopt, 0, 3, 0, 0), "mode must be 0..2, got 3"},
        {make("K1ABC", std::nullopt, 0, 1, 2, 0), "segment must be 0..mode (1), got 2"},
        {make("K1ABC", std::nullopt, 0, 0, 0, 1 << 16), "codec_id out of range: 65536"},
        {make("K1ABC", std::nullopt, 0, 0, 0, 0, 0), "version must be one of (1, 2), got 0"},
        {make("K1ABC", std::nullopt, 0, 0, 0, 0, 3), "version must be one of (1, 2), got 3"},
        {make("K1ABC", std::nullopt, 0, 0, 0, 0, 2, 1), "reserved must be 0, got 1"},
        // The reference checks the fields in this order, the grid last.
        {make("k1abc", "XX99", -1, 3, 0, 0), "callsign must be 1-8 characters of [A-Z0-9/], got 'k1abc'"},
        {make("K1ABC", "XX99", 0, 3, 0, 0), "mode must be 0..2, got 3"},
    };
    for (const auto& c : cases)
        check::equal(refusal([&] { hdr::pack(c.h); }), std::string(c.text), std::string("pack refuses: ") + c.text);
    // A caller's own texts replace the C++ formatting (the binding's FieldText).
    hdr::FieldText t;
    t.mode = "3.0";
    check::equal(refusal([&] { hdr::pack(make("K1ABC", std::nullopt, 0, 3, 0, 0), t); }),
                 std::string("mode must be 0..2, got 3.0"), "FieldText");
}

static void test_version_fallback() {
    auto v1 = H;
    v1.version = 1;
    const auto coded = hdr::encode(v1);
    std::vector<double> llr(coded.size());
    for (std::size_t e = 0; e < coded.size(); ++e) llr[e] = 8.0 * (1.0 - 2.0 * coded[e]);
    check::is_true(!hdr::decode(llr), "a version 1 header is refused by default");
    const std::array<int, 2> both{1, 2};
    check::is_true(hdr::decode(llr, 8, both) == std::optional<hdr::HeaderFields>(v1), "and accepted on request");
}

int main() {
    try {
        test_crc();
        test_layout();
        test_pack_digests();
        test_unpack_rejects();
        test_grid();
        test_pack_refusals();
        test_version_fallback();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return check::report("qrss_header");
}
