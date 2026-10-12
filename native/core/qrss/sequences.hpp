// SHA-256 bit streams of QRSSTVAE: the CE and L preambles, their
// reference symbols, the per-pass scrambler and the spread header's
// whitener. Port of sstvae/qrss/sequences.py (the reference).
//
// A format module. Every sequence is closed form over integers: block b
// is SHA256(domain || prefix || uint32_be(b)); bit i is bit (7 - i%8) of
// byte (i%256)/8 of block i/256 (MSB first); bit 0 maps to +1 and bit 1
// to -1 (decision D5). L takes the bits in pairs, the first to I and the
// second to Q, each scaled by 1/sqrt(2).
//
// The slot clock is here too, as in Python: q = floor(unix_utc(QH)/900),
// unix time ignoring leap seconds. Parsing ISO 8601 text is the CLI's job
// and lives with it.

#ifndef SSTVAE_QRSS_SEQUENCES_HPP
#define SSTVAE_QRSS_SEQUENCES_HPP

#include <complex>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace sstvae::qrss::sequences {

// The first `n` bits of SHA256(domain || prefix || u32be(b)), 0/1, MSB first.
std::vector<std::uint8_t> sha_bits(std::string_view domain, std::size_t n,
                                   std::span<const std::uint8_t> prefix = {});

// 0 -> +1, 1 -> -1 (int8 arithmetic, as numpy does: 1 - 2*b).
std::vector<std::int8_t> pm1(std::span<const std::uint8_t> bits);

// Bits 2i and 2i+1 to I and Q of symbol i, 0 -> +1, scaled by 1/sqrt(2).
// Throws std::invalid_argument for an odd count.
std::vector<std::complex<double>> qpsk(std::span<const std::uint8_t> bits);

// The CE preamble: N_PRE = 660 values of +-1.
const std::vector<std::int8_t>& preamble_ce();

// CE reference symbols; reference m takes bit m of one stream for the
// whole frame, so a shorter frame's references are a prefix.
std::vector<std::int8_t> references_ce(std::size_t n_ref);

// The L preamble: L_N_PRE = 330 QPSK symbols.
const std::vector<std::complex<double>>& preamble_l();

// L reference symbols; reference m takes bits 2m and 2m+1.
std::vector<std::complex<double>> references_l(std::size_t n_ref);

// Sign flips of the spread header, by data symbol. Not keyed by q.
std::vector<std::int8_t> spread_whitener(std::size_t n);

// Per-pass sign flips c_q, by data latent in air order, keyed by the
// slot's quarter-hour count as uint64_be.
std::vector<std::int8_t> scrambler(std::uint64_t q, std::size_t n);

// q for a unix time exactly on a quarter hour; std::invalid_argument
// otherwise ("not on a quarter hour").
std::int64_t quarter_hour_count(std::int64_t unix_seconds);

// Unix seconds of quarter hour q.
constexpr std::int64_t quarter_hour_start(std::int64_t q) { return 900 * q; }

}  // namespace sstvae::qrss::sequences

#endif  // SSTVAE_QRSS_SEQUENCES_HPP
