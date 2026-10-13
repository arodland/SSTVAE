// scipy.ndimage.median_filter on a 1-D array, mode "reflect", as
// acquire.spectral_shape calls it (`ndimage.median_filter(lvl, size=w,
// mode="reflect")`, w odd).
//
// Bit-exact by construction and by method: a median of an odd window is
// one of its elements, and the selection is scipy 1.18.1's own -- the
// 1-D fast path (scipy/ndimage/src/_rank_filter_1d.cpp: a double heap,
// "Mediator", over a circular window, primed with the reflected head of
// the array exactly as scipy primes it), so even which of two equal
// values (0.0 and -0.0) is returned, and what a NaN does to the order,
// are scipy's. Where scipy leaves the fast path (a window wider than the
// array would need, lim2 < 0 with more than one element) its generic
// filter selects the same rank from the reflected window; that is done
// here with a sort of the reflected window, which agrees for ordered
// (NaN-free) data.
//
// "reflect" is ndimage's half-sample symmetric extension
// (d c b a | a b c d | d c b a). Thread-safe: no state.

#ifndef SSTVAE_MATH_NDFILTER_HPP
#define SSTVAE_MATH_NDFILTER_HPP

#include <span>
#include <vector>

namespace sstvae::math {

// scipy.ndimage.median_filter(x, size=size, mode="reflect") of 1-D x.
// size must be odd and positive (std::invalid_argument otherwise: the
// receiver only uses odd sizes, and an even size is a rank filter scipy
// sends to minimum/maximum_filter or an off-centre window -- not ported).
// Empty x gives an empty result.
std::vector<double> median_filter(std::span<const double> x, int size);
std::vector<float> median_filter(std::span<const float> x, int size);

}  // namespace sstvae::math

#endif  // SSTVAE_MATH_NDFILTER_HPP
