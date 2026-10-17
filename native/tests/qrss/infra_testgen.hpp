// Deterministic test inputs for the infrastructure ctests (test_half,
// test_npsum, test_rational), reproducible from numpy so the digests they
// are checked against could be computed in Python:
//
//     def splitmix(seed, n):          # outputs 1..n, uint64 arithmetic wraps
//         s = np.uint64(seed) + np.arange(1, n + 1, dtype=np.uint64) * np.uint64(0x9E3779B97F4A7C15)
//         z = (s ^ (s >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
//         z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
//         return z ^ (z >> np.uint64(31))
//
//     def unit(z):                    # [0, 1), exact
//         return (z >> np.uint64(11)).astype(np.float64) * 2.0 ** -53
//
// (splitmix64; a hash of a counter, not a random generator in the
// layering rule's sense -- and this is test code.)

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace sstvae::qrss_testgen {

inline std::vector<std::uint64_t> splitmix(std::uint64_t seed, std::size_t n) {
    std::vector<std::uint64_t> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        std::uint64_t z = seed + static_cast<std::uint64_t>(i + 1) * 0x9E3779B97F4A7C15ULL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        out[i] = z ^ (z >> 31);
    }
    return out;
}

inline double unit(std::uint64_t z) { return static_cast<double>(z >> 11) * 0x1p-53; }

}  // namespace sstvae::qrss_testgen
