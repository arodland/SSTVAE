#include "qrss/crc32.hpp"

#include <array>
#include <cstddef>

namespace sstvae::qrss {

namespace {

// Slicing-by-8 (Intel's): T[0] is the byte table, T[s][i] = T[0] applied
// to T[s-1][i] shifted one more byte; eight bytes per step.
constexpr std::array<std::array<std::uint32_t, 256>, 8> make_tables() {
    std::array<std::array<std::uint32_t, 256>, 8> t{};
    for (std::uint32_t i = 0; i < 256; ++i) {
        std::uint32_t c = i;
        for (int k = 0; k < 8; ++k) c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
        t[0][i] = c;
    }
    for (std::uint32_t i = 0; i < 256; ++i)
        for (std::size_t s = 1; s < 8; ++s) t[s][i] = (t[s - 1][i] >> 8) ^ t[0][t[s - 1][i] & 0xffu];
    return t;
}

constexpr auto T = make_tables();
static_assert(T[0][1] == 0x77073096u && T[0][255] == 0x2D02EF8Du, "CRC-32 table");

std::uint32_t le32(const std::uint8_t* p) {
    return std::uint32_t(p[0]) | std::uint32_t(p[1]) << 8 | std::uint32_t(p[2]) << 16 | std::uint32_t(p[3]) << 24;
}

}  // namespace

std::uint32_t crc32(std::span<const std::uint8_t> data, std::uint32_t value) {
    std::uint32_t c = ~value;
    const std::uint8_t* p = data.data();
    std::size_t n = data.size();
    for (; n >= 8; p += 8, n -= 8) {
        const std::uint32_t a = le32(p) ^ c, b = le32(p + 4);
        c = T[7][a & 0xffu] ^ T[6][(a >> 8) & 0xffu] ^ T[5][(a >> 16) & 0xffu] ^ T[4][a >> 24] ^
            T[3][b & 0xffu] ^ T[2][(b >> 8) & 0xffu] ^ T[1][(b >> 16) & 0xffu] ^ T[0][b >> 24];
    }
    for (; n > 0; ++p, --n) c = T[0][(c ^ *p) & 0xffu] ^ (c >> 8);
    return ~c;
}

std::uint32_t crc32(std::string_view data, std::uint32_t value) {
    return crc32(std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(data.data()), data.size()),
                 value);
}

}  // namespace sstvae::qrss
