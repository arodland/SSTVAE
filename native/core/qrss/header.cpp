#include "qrss/header.hpp"

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "beacon/beacon.hpp"
#include "qrss/polar.hpp"

namespace sstvae::qrss::header {

namespace {

constexpr std::size_t UK = static_cast<std::size_t>(config::N_INFO_BITS);
constexpr std::size_t BODY = 126;   // everything the CRC covers

struct Offsets {
    std::array<std::pair<int, int>, FIELDS.size()> span{};
    constexpr Offsets() {
        int o = 0;
        for (std::size_t i = 0; i < FIELDS.size(); ++i) {
            span[i] = {o, o + FIELDS[i].width};
            o += FIELDS[i].width;
        }
    }
};
constexpr Offsets OFFSETS{};

constexpr std::pair<int, int> span_of(std::size_t i) { return OFFSETS.span[i]; }
constexpr std::size_t CALLSIGN = 0, GRID = 1, PICTURE_ID = 2, MODE = 3, SEGMENT = 4, CODEC_ID = 5,
                      VERSION = 6, RESERVED = 7, CRC = 8;
static_assert(span_of(CRC).first == static_cast<int>(BODY) && span_of(CRC).second == 142 &&
                  config::N_INFO_BITS == 142,
              "142 header bits, the CRC last (header._OFFSET)");

// beacon._int_to_bits: `width` bits of `value`, MSB first.
void append_bits(std::vector<std::uint8_t>& out, std::int64_t value, int width) {
    for (int k = width - 1; k >= 0; --k)
        out.push_back(static_cast<std::uint8_t>((static_cast<std::uint64_t>(value) >> k) & 1u));
}

// beacon._bits_to_int.
std::int64_t bits_to_int(std::span<const std::uint8_t> bits) {
    std::int64_t v = 0;
    for (std::uint8_t b : bits) v = (v << 1) | static_cast<std::int64_t>(b);
    return v;
}

std::int64_t field(std::span<const std::uint8_t> bits, std::size_t i) {
    const auto [a, b] = span_of(i);
    return bits_to_int(bits.subspan(static_cast<std::size_t>(a), static_cast<std::size_t>(b - a)));
}

// Python's str() of FORMAT_VERSIONS, a tuple.
std::string versions_text() {
    std::string s = "(";
    for (std::size_t i = 0; i < config::FORMAT_VERSIONS.size(); ++i) {
        if (i) s += ", ";
        s += std::to_string(config::FORMAT_VERSIONS[i]);
    }
    if (config::FORMAT_VERSIONS.size() == 1) s += ",";
    return s + ")";
}

std::string num(const std::optional<std::string>& text, std::int64_t v) {
    return text ? *text : std::to_string(v);
}

char ascii_upper(char c) { return (c >= 'a' && c <= 'z') ? static_cast<char>(c - 'a' + 'A') : c; }

}  // namespace

std::pair<int, int> offset(std::string_view name) {
    for (std::size_t i = 0; i < FIELDS.size(); ++i)
        if (FIELDS[i].name == name) return span_of(i);
    throw std::invalid_argument("no header field named " + std::string(name));
}

std::string py_repr(std::string_view s) {
    const bool has_single = s.find('\'') != std::string_view::npos;
    const bool has_double = s.find('"') != std::string_view::npos;
    const char q = (has_single && !has_double) ? '"' : '\'';
    std::string out(1, q);
    for (char ch : s) {
        const auto c = static_cast<unsigned char>(ch);
        if (ch == q || ch == '\\') {
            out += '\\';
            out += ch;
        } else if (ch == '\t') {
            out += "\\t";
        } else if (ch == '\n') {
            out += "\\n";
        } else if (ch == '\r') {
            out += "\\r";
        } else if (c < 0x20 || c == 0x7f) {
            char buf[8];
            std::snprintf(buf, sizeof buf, "\\x%02x", static_cast<unsigned>(c));
            out += buf;
        } else {
            out += ch;
        }
    }
    out += q;
    return out;
}

bool valid_callsign(std::string_view call) {
    if (call.empty() || call.size() > static_cast<std::size_t>(CALL_CHARS)) return false;
    for (char c : call)
        if (!((c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '/')) return false;
    return true;
}

std::vector<std::uint8_t> crc16(std::span<const std::uint8_t> bits) {
    std::vector<int> b(bits.size());
    for (std::size_t i = 0; i < bits.size(); ++i) {
        if (bits[i] > 1) throw std::invalid_argument("crc16 takes bits of 0 or 1");
        b[i] = bits[i];
    }
    const std::vector<int> c = beacon::crc16(b);
    return std::vector<std::uint8_t>(c.begin(), c.end());
}

int grid_encode(std::optional<std::string_view> g, const std::optional<std::string>& text) {
    if (!g) return GRID_NONE;
    std::string u(*g);
    std::transform(u.begin(), u.end(), u.begin(), ascii_upper);
    const bool ok = u.size() == 4 && u[0] >= 'A' && u[0] <= 'R' && u[1] >= 'A' && u[1] <= 'R' &&
                    u[2] >= '0' && u[2] <= '9' && u[3] >= '0' && u[3] <= '9';
    if (!ok)
        throw std::invalid_argument("grid must be a 4-character Maidenhead locator, got " +
                                    (text ? *text : py_repr(*g)));
    const int f1 = u[0] - 'A', f2 = u[1] - 'A';
    return ((f1 * 18 + f2) * 10 + (u[2] - '0')) * 10 + (u[3] - '0');
}

std::optional<std::string> grid_decode(std::int64_t v) {
    if (v == GRID_NONE) return std::nullopt;
    if (!(0 <= v && v < N_GRID_CODES)) throw std::invalid_argument("invalid grid code " + std::to_string(v));
    const int f = static_cast<int>(v / 100), d = static_cast<int>(v % 100);
    std::string s;
    s += static_cast<char>('A' + f / 18);
    s += static_cast<char>('A' + f % 18);
    s += static_cast<char>('0' + d / 10);
    s += static_cast<char>('0' + d % 10);
    return s;
}

void check_fields(const HeaderFields& h, const FieldText& text) {
    if (!valid_callsign(h.callsign))
        throw std::invalid_argument("callsign must be 1-8 characters of [A-Z0-9/], got " +
                                    (text.callsign ? *text.callsign : py_repr(h.callsign)));
    if (!(0 <= h.picture_id && h.picture_id < (std::int64_t(1) << 32)))
        throw std::invalid_argument("picture_id out of range: " + num(text.picture_id, h.picture_id));
    if (!(0 <= h.codec_id && h.codec_id < (std::int64_t(1) << 16)))
        throw std::invalid_argument("codec_id out of range: " + num(text.codec_id, h.codec_id));
    if (!(0 <= h.mode && h.mode <= MAX_MODE))
        throw std::invalid_argument("mode must be 0.." + std::to_string(MAX_MODE) + ", got " +
                                    num(text.mode, h.mode));
    if (!(0 <= h.segment && h.segment <= h.mode))
        throw std::invalid_argument("segment must be 0..mode (" + num(text.mode, h.mode) + "), got " +
                                    num(text.segment, h.segment));
    if (std::find(config::FORMAT_VERSIONS.begin(), config::FORMAT_VERSIONS.end(), h.version) ==
        config::FORMAT_VERSIONS.end())
        throw std::invalid_argument("version must be one of " + versions_text() + ", got " +
                                    num(text.version, h.version));
    if (h.reserved != 0)
        throw std::invalid_argument("reserved must be 0, got " + num(text.reserved, h.reserved));
}

std::vector<std::uint8_t> pack(const HeaderFields& h, const FieldText& text) {
    check_fields(h, text);
    const std::vector<int> codes = beacon::callsign_to_codes(h.callsign);
    const int grid = grid_encode(h.grid ? std::optional<std::string_view>(*h.grid) : std::nullopt, text.grid);
    std::vector<std::uint8_t> bits;
    bits.reserve(UK);
    for (int c : codes) append_bits(bits, c, 6);
    append_bits(bits, grid, FIELDS[GRID].width);
    append_bits(bits, h.picture_id, FIELDS[PICTURE_ID].width);
    append_bits(bits, h.mode, FIELDS[MODE].width);
    append_bits(bits, h.segment, FIELDS[SEGMENT].width);
    append_bits(bits, h.codec_id, FIELDS[CODEC_ID].width);
    append_bits(bits, h.version, FIELDS[VERSION].width);
    append_bits(bits, h.reserved, FIELDS[RESERVED].width);
    const std::vector<std::uint8_t> crc = crc16(bits);
    bits.insert(bits.end(), crc.begin(), crc.end());
    return bits;
}

std::optional<HeaderFields> unpack(std::span<const std::uint8_t> bits, std::span<const int> versions) {
    if (bits.size() != UK) return std::nullopt;
    for (std::uint8_t b : bits)
        if (b > 1) return std::nullopt;
    const std::vector<std::uint8_t> crc = crc16(bits.first(BODY));
    if (!std::equal(crc.begin(), crc.end(), bits.begin() + static_cast<std::ptrdiff_t>(BODY)))
        return std::nullopt;
    const std::int64_t version = field(bits, VERSION), reserved = field(bits, RESERVED);
    const std::int64_t mode = field(bits, MODE), segment = field(bits, SEGMENT);
    const bool known = std::find(versions.begin(), versions.end(), version) != versions.end();
    if (!known || reserved != 0 || mode > MAX_MODE || segment > mode) return std::nullopt;
    const int a = span_of(CALLSIGN).first;
    std::string call;
    for (int i = 0; i < CALL_CHARS; ++i) {
        const auto code = bits_to_int(bits.subspan(static_cast<std::size_t>(a + 6 * i), 6));
        call += beacon::ALPHABET[static_cast<std::size_t>(code)];
    }
    while (!call.empty() && call.back() == ' ') call.pop_back();   // rstrip(" ")
    if (!valid_callsign(call)) return std::nullopt;                 // also an internal space
    const std::int64_t g = field(bits, GRID);
    if (g != GRID_NONE && g >= N_GRID_CODES) return std::nullopt;  // grid_decode would refuse
    HeaderFields h;
    h.callsign = std::move(call);
    h.grid = grid_decode(g);
    h.picture_id = field(bits, PICTURE_ID);
    h.mode = mode;
    h.segment = segment;
    h.codec_id = field(bits, CODEC_ID);
    h.version = version;
    h.reserved = reserved;
    return h;
}

std::vector<std::uint8_t> encode(const HeaderFields& h, const FieldText& text) {
    return polar::polar_encode(pack(h, text));
}

std::optional<HeaderFields> decode(std::span<const double> llr, std::size_t list_size,
                                   std::span<const int> versions) {
    const auto bits = polar::polar_decode(llr, list_size, [versions](std::span<const std::uint8_t> b) {
        return unpack(b, versions).has_value();
    });
    if (!bits) return std::nullopt;
    return unpack(*bits, versions);
}

}  // namespace sstvae::qrss::header
