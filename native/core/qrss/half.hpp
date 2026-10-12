// IEEE 754 binary16 ("half", numpy's float16) as bit patterns, converted
// exactly as numpy converts them: npy_double_to_half / npy_float_to_half
// and npy_half_to_double / npy_half_to_float (numpy/_core/src/common/
// half_private.hpp, FromDoubleBits and friends).
//
// The one rule that matters: **double -> half is one rounding, taken
// directly from the double** (round half to even). Going through float32
// first rounds twice, and 5.5e-5 of values then land one ulp away -- about
// eight latents of a mode-C picture, which is a different picture ID
// (picture.prepare_segment hashes the fp16 values). numpy's software path
// does the single rounding on the double's bits; so does this.
//
// Hand-written on purpose: MSVC has no _Float16, and the compilers that do
// differ on how they convert NaNs. Everything here is integer arithmetic
// on bit patterns, so it is the same on every platform.
//
// NaNs keep their payload as numpy's software path keeps it: half -> double
// shifts the 10 significand bits to the top of the double's 52; double ->
// half keeps the top 10 and, if that leaves zero, sets the lowest so the
// result is still a NaN. (numpy's float32 -> float16 cast may go through
// F16C hardware instead, which quiets a signalling NaN differently; only a
// NaN's payload can differ, never a number. Nothing in sstvae/qrss converts
// float32 to float16.)

#ifndef SSTVAE_QRSS_HALF_HPP
#define SSTVAE_QRSS_HALF_HPP

#include <cstdint>
#include <span>
#include <vector>

namespace sstvae::qrss::half {

// A float16 value's bits, as a distinct type so a vector of them is not
// mistaken for uint16 data (npz maps it to '<f2', not '<u2').
struct Half {
    std::uint16_t bits = 0;
    friend constexpr bool operator==(Half a, Half b) { return a.bits == b.bits; }
};
static_assert(sizeof(Half) == 2, "Half must be exactly the 2 bytes numpy stores");

// numpy's npy_doublebits_to_halfbits: round half to even, overflow to
// +-inf, underflow through the subnormals to +-0.
std::uint16_t from_double(double x);
// numpy's npy_floatbits_to_halfbits: the same, from a float32.
std::uint16_t from_float(float x);
// Exact (every half is a double and a float).
double to_double(std::uint16_t h);
float to_float(std::uint16_t h);

// Elementwise, as x.astype(np.float16) and h.astype(np.float64).
std::vector<std::uint16_t> from_double(std::span<const double> x);
std::vector<double> to_double(std::span<const std::uint16_t> h);

inline Half to_half(double x) { return Half{from_double(x)}; }
inline double to_double(Half h) { return to_double(h.bits); }

}  // namespace sstvae::qrss::half

#endif  // SSTVAE_QRSS_HALF_HPP
