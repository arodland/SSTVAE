// The polar (2048, 142) code of the QRSSTVAE header (design section 2.6).
// Port of sstvae/qrss/polar.py (the reference).
//
// A format module: the encoder and INFO_SET define what goes on the air.
//
//     mother code   x = u . F^(x)11 (mod 2), F = [[1, 0], [1, 1]],
//                   natural order, no bit reversal: x_j = XOR_{i superset of j} u_i
//     info bits     header bit t -> u[INFO_SET[t]]; every other u is frozen at 0
//     rate matching circular repetition, coded bit e = x[e mod 2048], e < 2474
//
// INFO_SET is committed literal data, generated into qrss/config.hpp from
// the Python tuple, with its SHA-256 beside it.
//
// The decoder (CA-SCL, list 8) is not on-air format but lives here, as it
// does in Python, and reproduces the reference's arithmetic operation for
// operation: min-sum f, exact g, the LLR-based path metric
// PM += log(1 + exp(-(1 - 2u) lambda)) computed as numpy's logaddexp, the
// rate-0 shortcut summed as numpy's pairwise sum, and the list pruned by a
// stable sort in numpy's order (NaN last). Decisions and path metrics
// therefore agree with Python bit for bit wherever the two share a libm.
//
// LLR convention: llr = log P(b=0)/P(b=1), positive means bit 0.

#ifndef SSTVAE_QRSS_POLAR_HPP
#define SSTVAE_QRSS_POLAR_HPP

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "qrss/config.hpp"

namespace sstvae::qrss::polar {

inline constexpr int N = config::POLAR_N;        // 2048 mother-code length
inline constexpr int K = config::N_INFO_BITS;    // 142 information bits, CRC included
inline constexpr int E = config::N_HDR_BITS;     // 2474 coded bits on the air
inline constexpr int N_STAGES = 11;
static_assert((1 << N_STAGES) == N, "N is 2^N_STAGES");
static_assert(static_cast<int>(config::POLAR_INFO_SET.size()) == K, "one information index per header bit");

// INFO_SET, ascending (config::POLAR_INFO_SET).
inline constexpr const auto& INFO_SET = config::POLAR_INFO_SET;

// True when INFO_SET hashes (as big-endian uint16) to POLAR_INFO_SET_SHA256
// and is strictly ascending inside [0, N): the Python module's import-time
// asserts, checked once.
bool info_set_ok();

// Coded bit e carries mother bit e mod N.
constexpr int mother_index(int e) { return e % N; }

// v . F^(x)n (mod 2), natural order, in place over consecutive rows of
// length n (x.size() a multiple of n). Self-inverse. As numpy's reshape in
// the reference, every h = n/2, n/4, ..., 1 must divide n into 2h-blocks;
// std::invalid_argument otherwise. Any byte values, XORed.
void polar_transform_rows(std::span<std::uint8_t> x, std::size_t n);

// One row.
std::vector<std::uint8_t> polar_transform(std::span<const std::uint8_t> v);

// Mother codeword (N) -> coded bits (E) by circular repetition, any
// trivially copyable element; `x` needs at least N entries.
template <typename T>
void rate_match_into(std::span<const T> x, std::span<T> out) {
    if (x.size() < static_cast<std::size_t>(N))
        throw std::invalid_argument("rate_match needs " + std::to_string(N) + " mother bits, got " +
                                    std::to_string(x.size()));
    if (out.size() != static_cast<std::size_t>(E))
        throw std::invalid_argument("rate_match writes " + std::to_string(E) + " coded bits");
    for (int e = 0; e < E; ++e) out[static_cast<std::size_t>(e)] = x[static_cast<std::size_t>(mother_index(e))];
}

std::vector<std::uint8_t> rate_match(std::span<const std::uint8_t> x);

// Coded LLRs (E) -> mother LLRs (N): repeated bits add, in coded order from
// 0.0 (np.bincount). std::invalid_argument
// "expected 2474 coded LLRs, got shape (n,)" for another length.
std::vector<double> fold(std::span<const double> llr);

// 142 header bits (table order, 0/1) -> 2474 coded bits, 0/1.
// std::invalid_argument for another length or a value other than 0/1.
std::vector<std::uint8_t> polar_encode(std::span<const std::uint8_t> info);

// The surviving list paths of a decode, most likely first.
struct ListResult {
    std::size_t m = 0;                  // number of paths (<= list size)
    std::vector<std::uint8_t> info;     // m x K, row-major
    std::vector<double> pm;             // m path metrics, ascending
    std::span<const std::uint8_t> row(std::size_t i) const {
        return std::span<const std::uint8_t>(info).subspan(i * K, K);
    }
};

// SCL over coded LLRs (E). No CRC or field check. std::invalid_argument
// "list_size must be >= 1", the fold() shape refusal, or
// "LLRs must be finite" (after folding).
ListResult decode_list(std::span<const double> llr, std::size_t list_size = 8);

// Acceptance test on 142 header bits.
using Accept = std::function<bool(std::span<const std::uint8_t>)>;

// Coded LLRs -> the most likely list path that `accept` passes, or
// nullopt. An empty `accept` is the header's own check
// (header::unpack(bits) with the default versions), as in Python.
std::optional<std::vector<std::uint8_t>> polar_decode(std::span<const double> llr,
                                                      std::size_t list_size = 8,
                                                      const Accept& accept = {});

}  // namespace sstvae::qrss::polar

#endif  // SSTVAE_QRSS_POLAR_HPP
