// The QRSSTVAE header: 142 bits, polar coded to 2474 (design section 2.6).
// Port of sstvae/qrss/header.py (the reference).
//
// A format module. Fields, in table order, each MSB first:
//
//     bits   0-47   callsign      8 x 6 bits, SSTVAE's beacon alphabet, space padded
//     bits  48-62   grid          ((F1*18 + F2)*10 + D1)*10 + D2; 32767 = none
//     bits  63-94   picture ID    32 bits
//     bits  95-96   mode          A = 0, B = 1, C = 2
//     bits  97-98   segment       0..mode, the latent group
//     bits  99-114  codec ID      16 bits
//     bits 115-118  version       FORMAT_VERSION (2; 1 was sent only in tests)
//     bits 119-125  reserved      0
//     bits 126-141  CRC           crc16(bits[0:126])
//
// The CRC is SSTVAE's beacon CRC (beacon::crc16), bit for bit -- check
// value 0xA69D over ASCII "123456789", not CRC-16/CCITT-FALSE's 0x29B1
// (decision D8). It is reused, not re-implemented, and must not be "fixed".
//
// A decode is accepted only when the CRC matches and the version is one
// asked for (FORMAT_VERSION by default), reserved = 0, mode <= 2,
// segment <= mode, the grid code is valid and the callsign is a valid
// QRSS call ([A-Z0-9/], 1 to 8 characters, space padded) (decision D9).

#ifndef SSTVAE_QRSS_HEADER_HPP
#define SSTVAE_QRSS_HEADER_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "qrss/qrss_config.hpp"

namespace sstvae::qrss::header {

inline constexpr int CALL_CHARS = 8;
inline constexpr int GRID_NONE = 32767;
inline constexpr int MAX_MODE = 2;
inline constexpr int N_GRID_CODES = 18 * 18 * 100;   // 32400

// Field widths in table order (header._FIELDS); offsets follow.
struct Field {
    std::string_view name;
    int width;
};
inline constexpr std::array<Field, 9> FIELDS{{
    {"callsign", 48}, {"grid", 15}, {"picture_id", 32}, {"mode", 2}, {"segment", 2},
    {"codec_id", 16}, {"version", 4}, {"reserved", 7}, {"crc", 16},
}};

// [start, end) of a field by name (header._OFFSET); std::invalid_argument
// for an unknown name.
std::pair<int, int> offset(std::string_view name);

// header.HeaderFields. The integers are int64 so that a value a caller
// gets wrong can be held, and refused with its own value in the message.
struct HeaderFields {
    std::string callsign;
    std::optional<std::string> grid;
    std::int64_t picture_id = 0;
    std::int64_t mode = 0;
    std::int64_t segment = 0;
    std::int64_t codec_id = 0;
    std::int64_t version = config::FORMAT_VERSION;
    std::int64_t reserved = 0;
    bool operator==(const HeaderFields&) const = default;
};

// How a refusal prints each field, when the caller knows better than the
// C++ formatting (repr-style quotes for the strings, decimal for the
// integers). The Python binding fills these with Python's own repr()/str()
// so that its ValueError texts are the reference's exactly.
struct FieldText {
    std::optional<std::string> callsign, grid, picture_id, mode, segment, codec_id, version, reserved;
};

// The default `versions` of unpack and decode: (FORMAT_VERSION,).
inline constexpr std::array<int, 1> DEFAULT_VERSIONS{config::FORMAT_VERSION};

// Python's repr() of an ASCII string (quote choice, \\, \n, \r, \t, \xNN);
// bytes >= 0x80 are passed through as they are.
std::string py_repr(std::string_view s);

// True for 1-8 characters of [A-Z0-9/] (header._CALL_RE.fullmatch).
bool valid_callsign(std::string_view call);

// SSTVAE's beacon CRC over 0/1 bits -> 16 bits, MSB first.
// std::invalid_argument for a value other than 0/1.
std::vector<std::uint8_t> crc16(std::span<const std::uint8_t> bits);

// 4-character Maidenhead locator -> 15-bit code; nullopt -> 32767. ASCII
// case-insensitive. std::invalid_argument
// "grid must be a 4-character Maidenhead locator, got <repr>" otherwise;
// `text` replaces <repr> when given.
int grid_encode(std::optional<std::string_view> g, const std::optional<std::string>& text = std::nullopt);

// 15-bit code -> locator; 32767 -> nullopt. std::invalid_argument
// "invalid grid code <v>" for any other code outside 0..32399 (the
// 32400..32766 codes name no square).
std::optional<std::string> grid_decode(std::int64_t v);

// header._check_fields: refuses, in the reference's order and with its
// texts, anything unpack would reject.
void check_fields(const HeaderFields& h, const FieldText& text = {});

// HeaderFields -> 142 bits (0/1, table order, CRC last). Refuses
// (std::invalid_argument) anything unpack would reject.
std::vector<std::uint8_t> pack(const HeaderFields& h, const FieldText& text = {});

// 142 bits -> HeaderFields, or nullopt if the length, a value other than
// 0/1, the CRC or any field check fails (the version must be in `versions`).
std::optional<HeaderFields> unpack(std::span<const std::uint8_t> bits,
                                   std::span<const int> versions = DEFAULT_VERSIONS);

// HeaderFields -> 2474 coded bits. Bit e rides header symbol e as +1 for 0.
std::vector<std::uint8_t> encode(const HeaderFields& h, const FieldText& text = {});

// 2474 coded-bit LLRs (log P(0)/P(1)) -> HeaderFields, or nullopt: the
// most likely list path that unpack(bits, versions) accepts. For soft
// combining pass the sum of several passes' LLRs; `versions` widens the
// version check (the block-alone fallback for version 1 recordings).
std::optional<HeaderFields> decode(std::span<const double> llr, std::size_t list_size = 8,
                                   std::span<const int> versions = DEFAULT_VERSIONS);

}  // namespace sstvae::qrss::header

#endif  // SSTVAE_QRSS_HEADER_HPP
