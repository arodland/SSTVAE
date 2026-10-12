#include "qrss/sequences.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "qrss/config.hpp"
#include "util/sha256.hpp"

namespace sstvae::qrss::sequences {

namespace {

constexpr std::size_t BITS_PER_BLOCK = 256;

std::vector<std::complex<double>> qpsk_of(const std::vector<std::uint8_t>& bits) { return qpsk(bits); }

}  // namespace

std::vector<std::uint8_t> sha_bits(std::string_view domain, std::size_t n,
                                   std::span<const std::uint8_t> prefix) {
    std::vector<std::uint8_t> out(n);
    const std::size_t n_blocks = (n + BITS_PER_BLOCK - 1) / BITS_PER_BLOCK;
    util::Sha256 head;
    head.update(domain);
    head.update(prefix);
    for (std::size_t b = 0; b < n_blocks; ++b) {
        util::Sha256 h = head;
        const std::uint8_t ctr[4] = {
            static_cast<std::uint8_t>(b >> 24), static_cast<std::uint8_t>(b >> 16),
            static_cast<std::uint8_t>(b >> 8), static_cast<std::uint8_t>(b)};
        h.update(std::span<const std::uint8_t>(ctr, 4));
        const util::Sha256Digest d = h.digest();
        const std::size_t i0 = b * BITS_PER_BLOCK;
        const std::size_t i1 = std::min(n, i0 + BITS_PER_BLOCK);
        for (std::size_t i = i0; i < i1; ++i) {
            const std::size_t j = i - i0;
            out[i] = static_cast<std::uint8_t>((d[j / 8] >> (7 - j % 8)) & 1u);
        }
    }
    return out;
}

std::vector<std::int8_t> pm1(std::span<const std::uint8_t> bits) {
    std::vector<std::int8_t> out(bits.size());
    for (std::size_t i = 0; i < bits.size(); ++i)
        out[i] = static_cast<std::int8_t>(1 - 2 * static_cast<std::int8_t>(bits[i]));
    return out;
}

std::vector<std::complex<double>> qpsk(std::span<const std::uint8_t> bits) {
    if (bits.size() % 2)
        throw std::invalid_argument("qpsk takes an even number of bits, not shape (" +
                                    std::to_string(bits.size()) + ",)");
    // numpy: pm1(b).astype(float64) / np.sqrt(2.0). sqrt is correctly
    // rounded, and +-1/sqrt(2) is one division, so this is exact.
    const double s = std::sqrt(2.0);
    std::vector<std::complex<double>> out(bits.size() / 2);
    for (std::size_t i = 0; i < out.size(); ++i) {
        const double re = static_cast<double>(1 - 2 * static_cast<int>(bits[2 * i])) / s;
        const double im = static_cast<double>(1 - 2 * static_cast<int>(bits[2 * i + 1])) / s;
        out[i] = {re, im};
    }
    return out;
}

const std::vector<std::int8_t>& preamble_ce() {
    static const std::vector<std::int8_t> v =
        pm1(sha_bits(config::PREAMBLE_DOMAIN, config::N_PRE));
    return v;
}

std::vector<std::int8_t> references_ce(std::size_t n_ref) {
    return pm1(sha_bits(config::REFERENCE_DOMAIN, n_ref));
}

const std::vector<std::complex<double>>& preamble_l() {
    static const std::vector<std::complex<double>> v =
        qpsk_of(sha_bits(config::L_PREAMBLE_DOMAIN, 2 * static_cast<std::size_t>(config::L_N_PRE)));
    return v;
}

std::vector<std::complex<double>> references_l(std::size_t n_ref) {
    return qpsk_of(sha_bits(config::L_REFERENCE_DOMAIN, 2 * n_ref));
}

std::vector<std::int8_t> spread_whitener(std::size_t n) {
    return pm1(sha_bits(config::SPREAD_DOMAIN, n));
}

std::vector<std::int8_t> scrambler(std::uint64_t q, std::size_t n) {
    std::uint8_t prefix[8];
    for (int i = 0; i < 8; ++i) prefix[i] = static_cast<std::uint8_t>(q >> (56 - 8 * i));
    return pm1(sha_bits(config::SCRAMBLE_DOMAIN, n, std::span<const std::uint8_t>(prefix, 8)));
}

std::int64_t quarter_hour_count(std::int64_t unix_seconds) {
    // Python: delta % 900 s nonzero -> refused; delta // 900 s (floor).
    std::int64_t r = unix_seconds % 900;
    if (r < 0) r += 900;
    if (r != 0) throw std::invalid_argument("not on a quarter hour");
    return (unix_seconds - r) / 900;
}

}  // namespace sstvae::qrss::sequences
