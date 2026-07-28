#include "golay/golay.hpp"

#include <bit>
#include <stdexcept>
#include <vector>

namespace sstvae::golay {
namespace {

// Remainder of value / POLY over GF(2).
std::uint32_t mod2div_remainder(std::uint32_t value) {
    const int bits = std::bit_width(value);
    for (int shift = bits - 12; shift >= 0; --shift) {
        if (value & (1u << (shift + 11))) value ^= static_cast<std::uint32_t>(POLY) << shift;
    }
    return value;
}

struct Tables {
    std::array<std::uint32_t, N_MESSAGES> codewords{};
    // (N_MESSAGES, N_BITS) of +/-1, bit 23 (MSB) first. 786 KiB, built
    // once; the Python reference holds the identical matrix and does the
    // identical dot product, so the port is a transcription rather than
    // a reinterpretation.
    std::vector<double> signs;

    Tables() : signs(static_cast<std::size_t>(N_MESSAGES) * N_BITS) {
        for (int m = 0; m < N_MESSAGES; ++m) {
            const std::uint32_t cw = encode(m);
            codewords[static_cast<std::size_t>(m)] = cw;
            for (int b = 0; b < N_BITS; ++b) {
                const int bit = static_cast<int>((cw >> (N_BITS - 1 - b)) & 1u);
                signs[static_cast<std::size_t>(m) * N_BITS + static_cast<std::size_t>(b)] =
                    1.0 - 2.0 * static_cast<double>(bit);
            }
        }
    }
};

const Tables& tables() {
    // Function-local static: built on first use, so there is no static
    // initialization order problem with the pybind11 module or with the
    // test binary's own globals.
    static const Tables t;
    return t;
}

}  // namespace

std::uint32_t encode(int data12) {
    if (data12 < 0 || data12 >= N_MESSAGES)
        throw std::out_of_range("golay::encode expects 12 bits");
    const std::uint32_t shifted = static_cast<std::uint32_t>(data12) << 11;
    const std::uint32_t cw23 = shifted | mod2div_remainder(shifted);
    const std::uint32_t parity = static_cast<std::uint32_t>(std::popcount(cw23)) & 1u;
    return (cw23 << 1) | parity;
}

void codeword_bits(int data12, std::span<int> out) {
    if (out.size() < N_BITS)
        throw std::invalid_argument("golay::codeword_bits needs 24 elements");
    const std::uint32_t cw = encode(data12);
    for (int b = 0; b < N_BITS; ++b)
        out[static_cast<std::size_t>(b)] =
            static_cast<int>((cw >> (N_BITS - 1 - b)) & 1u);
}

std::array<int, N_BITS> codeword_bits(int data12) {
    std::array<int, N_BITS> out{};
    codeword_bits(data12, out);
    return out;
}

int decode_soft(std::span<const double> soft) {
    if (soft.size() != N_BITS)
        throw std::invalid_argument("golay::decode_soft expects 24 soft values");
    const std::vector<double>& signs = tables().signs;
    int best = 0;
    double best_score = -1e300;
    for (int m = 0; m < N_MESSAGES; ++m) {
        const double* row = signs.data() + static_cast<std::size_t>(m) * N_BITS;
        double score = 0.0;
        for (int b = 0; b < N_BITS; ++b) score += row[b] * soft[static_cast<std::size_t>(b)];
        // Strict >, scanning upward: first maximum wins, as np.argmax does.
        if (score > best_score) {
            best_score = score;
            best = m;
        }
    }
    return best;
}

int min_distance() {
    int best = N_BITS;
    for (int m = 1; m < N_MESSAGES; ++m) {
        const int w = std::popcount(tables().codewords[static_cast<std::size_t>(m)]);
        if (w < best) best = w;
    }
    return best;
}

}  // namespace sstvae::golay
