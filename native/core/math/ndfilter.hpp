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
// filter (ni_filters.c NI_RankFilter) quickselects the rank from the
// reflected window in a double buffer; NI_Select is copied too, so the
// choice between equal values is again scipy's. The one case that cannot
// be scipy's to the bit is lim2 == 0 (window 2 n + 1 for n elements):
// scipy's fast path primes its heap with a read one past the array's end
// there, so its tie-breaking depends on whatever memory follows; the
// generic filter's answer is used, which is the same value (the sign of
// a zero aside) whenever scipy's heap survived that read.
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
