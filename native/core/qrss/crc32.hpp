// CRC-32 as zlib.crc32 computes it (ISO-HDLC / IEEE 802.3: reflected
// polynomial 0xEDB88320, initial and final XOR 0xFFFFFFFF), incremental
// the same way: crc32(b, crc32(a)) == crc32(a + b).
//
// Used by the beacon file (beaconfile.py's trailing CRC) and by npz, whose
// members carry it.

#ifndef SSTVAE_QRSS_CRC32_HPP
#define SSTVAE_QRSS_CRC32_HPP

#include <cstdint>
#include <span>
#include <string_view>

namespace sstvae::qrss {

// zlib.crc32(data, value): `value` is the CRC of everything before `data`.
std::uint32_t crc32(std::span<const std::uint8_t> data, std::uint32_t value = 0);
std::uint32_t crc32(std::string_view data, std::uint32_t value = 0);

class Crc32 {
public:
    void update(std::span<const std::uint8_t> data) { value_ = crc32(data, value_); }
    void update(std::string_view data) { value_ = crc32(data, value_); }
    std::uint32_t value() const { return value_; }

private:
    std::uint32_t value_ = 0;
};

}  // namespace sstvae::qrss

#endif  // SSTVAE_QRSS_CRC32_HPP
