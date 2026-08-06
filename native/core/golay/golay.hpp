// Extended Golay (24,12), systematic, with brute-force soft decoding.
//
// Port of sstvae/modem/golay.py. Encoding uses the cyclic (23,12)
// generator polynomial 0xC75 (x^11+x^10+x^6+x^5+x^4+x^2+1) plus an
// overall parity bit. The decoder correlates soft values against all
// 4096 codewords, which is trivially fast and gives true
// maximum-likelihood performance.

#pragma once

#include <array>
#include <cstdint>
#include <span>

namespace sstvae::golay {

inline constexpr int POLY = 0xC75;  // degree 11
inline constexpr int N_MESSAGES = 4096;
inline constexpr int N_BITS = 24;

// 12 info bits -> 24-bit codeword (data in high bits, parity last).
std::uint32_t encode(int data12);

// 24-bit codeword as 0/1, MSB first. `out` must have N_BITS room.
void codeword_bits(int data12, std::span<int> out);
std::array<int, N_BITS> codeword_bits(int data12);

// ML-decode 24 soft values (positive => bit 0) to the 12 info bits.
//
// Ties resolve to the lowest message index, matching np.argmax. That is
// not a detail worth preserving for its own sake, but it is observable
// from the golden vectors and free to keep, so the two implementations
// agree even on inputs where the answer is arbitrary.
int decode_soft(std::span<const double> soft);

// Minimum distance of the code (8). Used by tests.
int min_distance();

// Sign pattern (+1/-1, MSB first) for every message, N_MESSAGES rows of
// N_BITS each, row-major -- message `m`'s row is exactly what
// `decode_soft` correlates candidate soft values against. Exposed so a
// caller searching many hypotheses (beacon::_search_counter_chunk) can
// index straight into the precomputed table instead of re-deriving each
// hypothesis's codeword via codeword_bits() one at a time.
std::span<const double> signs_table();

}  // namespace sstvae::golay
