#include "qrss/crc32.hpp"

#include <array>
#include <cstddef>

namespace sstvae::qrss {

namespace {

constexpr std::array<std::uint32_t, 256> make_table() {
    std::array<std::uint32_t, 256> t{};
    for (std::uint32_t i = 0; i < 256; ++i) {
        std::uint32_t c = i;
        for (int k = 0; k < 8; ++k) c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
        t[i] = c;
    }
    return t;
}

constexpr std::array<std::uint32_t, 256> TABLE = make_table();
static_assert(TABLE[1] == 0x77073096u && TABLE[255] == 0x2D02EF8Du, "CRC-32 table");

}  // namespace

std::uint32_t crc32(std::span<const std::uint8_t> data, std::uint32_t value) {
    std::uint32_t c = ~value;
    for (const std::uint8_t b : data) c = TABLE[(c ^ b) & 0xffu] ^ (c >> 8);
    return ~c;
}

std::uint32_t crc32(std::string_view data, std::uint32_t value) {
    return crc32(std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(data.data()), data.size()),
                 value);
}

}  // namespace sstvae::qrss
