// The per-pass precoder of QRSSTVAE: sign flips, then a blockwise
// Walsh-Hadamard transform (spec 2.4, design 2.5). Port of
// sstvae/qrss/precoder.py (the reference).
//
// A format module. A segment's data latents in air order are split into
// blocks -- 64 at a time, then the binary decomposition of the remainder
// (50,600 = 790 x 64 + 32 + 8, decision D6) -- and each block goes out as
// H.(c_q * a), H the orthonormal Sylvester Hadamard matrix of the block's
// own size. H is symmetric and orthonormal, so the receiver undoes the
// precoder with the same transform followed by the same signs.
//
// Bit-exact against numpy, which is why the arithmetic is spelled out:
// the butterflies run h = 1, 2, 4, ... and replace (x, y) by (x + y,
// x - y); the block is then divided (not multiplied by a reciprocal) by
// sqrt(M). On fp16-grid latents every step is exact; on int8/20 latents
// only this order reproduces numpy's roundings.
//
// Arrays with leading batch axes are handled as rows: `v` holds whole
// rows of `n` values each (C order, the last axis), n == v.size() for a
// single vector.

#ifndef SSTVAE_QRSS_PRECODER_HPP
#define SSTVAE_QRSS_PRECODER_HPP

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace sstvae::qrss::precoder {

// precoder.BLOCK. Not yet emitted by tools/gen_qrss_config_header.py;
// tests/test_qrss_native_precoder.py checks it against the reference.
inline constexpr std::size_t BLOCK = 64;
static_assert((BLOCK & (BLOCK - 1)) == 0, "block_sizes takes the remainder's bits");

// [BLOCK] * (n / BLOCK) plus the binary decomposition of n % BLOCK,
// descending: 50600 -> 790 x 64, 32, 8.
std::vector<std::size_t> block_sizes(std::size_t n);

// A run of equal-size blocks: `count` blocks of `m` from `start`.
struct Run {
    std::size_t start;
    std::size_t m;
    std::size_t count;
};
// precoder._runs: the runs of block_sizes(n), largest blocks first.
std::vector<Run> runs(std::size_t n);

// The blockwise orthonormal WHT, in place over every row of `n`.
// std::invalid_argument when v.size() is not a whole number of rows.
void wht_inplace(std::span<double> v, std::size_t n);

// wht(v): a new array; self-inverse.
std::vector<double> wht(std::span<const double> v, std::size_t n);
inline std::vector<double> wht(std::span<const double> v) { return wht(v, v.size()); }

// x = WHT(c_q * a): air-order latents -> precoded data symbols.
std::vector<double> precode(std::span<const double> a, std::uint64_t q, std::size_t n);
inline std::vector<double> precode(std::span<const double> a, std::uint64_t q) {
    return precode(a, q, a.size());
}

// a = c_q * WHT(y): received data symbols -> air-order latents.
std::vector<double> unprecode(std::span<const double> y, std::uint64_t q, std::size_t n);
inline std::vector<double> unprecode(std::span<const double> y, std::uint64_t q) {
    return unprecode(y, q, y.size());
}

// How numpy's mean sums a block. np.add.reduce hands the add loop a whole
// block, which it sums pairwise, when the block's axis is the one its
// iterator runs innermost: the axis of smallest |stride|, which is the
// last for every C-ordered input. When a batch axis is laid out inside the
// last one (Fortran order, a transpose) that batch axis runs innermost
// instead, and each block is summed one element at a time from 0.0. The
// reference's last bit therefore depends on memory layout; the binding
// reads the layout and passes the order numpy will use.
enum class Summation { pairwise, sequential };

// The mean of each precoder block, broadcast back over the block, per row
// (numpy's mean: 0.0 + the sum, then / M).
std::vector<double> block_mean(std::span<const double> v, std::size_t n, Summation sum = Summation::pairwise);
inline std::vector<double> block_mean(std::span<const double> v) { return block_mean(v, v.size()); }

// The precoder block each of n data symbols belongs to.
std::vector<std::int64_t> block_index(std::size_t n);

}  // namespace sstvae::qrss::precoder

#endif  // SSTVAE_QRSS_PRECODER_HPP
