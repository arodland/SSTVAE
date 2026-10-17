// native/core/qrss/crc32 against zlib.crc32's values:
//
//   zlib.crc32(b"123456789")             == 0xCBF43926   (the CRC-32 check value)
//   zlib.crc32(bytes(range(256)) * 10)   == 0xC3B82B6C
//   zlib.crc32(b"a", 0xDEADBEEF)         == 0xD8DCEC14

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "qrss/crc32.hpp"
#include "qrss_check.hpp"

using namespace sstvae;

int main() {
    try {
        check::equal(qrss::crc32(std::string_view("")), std::uint32_t(0), "empty");
        check::equal(qrss::crc32(std::string_view("123456789")), std::uint32_t(0xCBF43926u), "check value");
        std::vector<std::uint8_t> blob;
        for (int r = 0; r < 10; ++r)
            for (int b = 0; b < 256; ++b) blob.push_back(static_cast<std::uint8_t>(b));
        check::equal(qrss::crc32(blob), std::uint32_t(0xC3B82B6Cu), "bytes(range(256)) * 10");
        check::equal(qrss::crc32(std::string_view("a"), 0xDEADBEEFu), std::uint32_t(0xD8DCEC14u), "with a start value");
        // Incremental, at every split point, both ways of feeding it.
        int bad = 0;
        for (std::size_t cut = 0; cut <= blob.size(); cut += 37) {
            const std::span<const std::uint8_t> all(blob);
            const std::uint32_t c = qrss::crc32(all.subspan(cut), qrss::crc32(all.first(cut)));
            qrss::Crc32 inc;
            inc.update(all.first(cut));
            inc.update(all.subspan(cut));
            bad += (c != 0xC3B82B6Cu) + (inc.value() != 0xC3B82B6Cu);
        }
        check::equal(bad, 0, "incremental");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return check::report("qrss_crc32");
}
