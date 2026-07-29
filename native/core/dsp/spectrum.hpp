// The numbers behind the waterfall display.
//
// Split out of the widget, and Qt-free, for the reason this project
// keeps rediscovering: the part with logic in it should be testable
// without the thing that makes it hard to test. `reduce_to_width` in
// particular has a property that is easy to get wrong and invisible
// once it is wrong -- see its comment.

#ifndef SSTVAE_DSP_SPECTRUM_HPP
#define SSTVAE_DSP_SPECTRUM_HPP

#include <cstddef>
#include <vector>

#include "config.hpp"

namespace sstvae::dsp {

inline constexpr int WATERFALL_NFFT = 1024;
inline constexpr double WATERFALL_BIN_HZ =
    static_cast<double>(config::FS) / WATERFALL_NFFT;
// An SSB receiver's passband; the signal lives inside it.
inline constexpr double WATERFALL_DISPLAY_HZ = 3000.0;
inline constexpr int WATERFALL_BINS =
    static_cast<int>(WATERFALL_DISPLAY_HZ / WATERFALL_BIN_HZ);

// Hann-windowed magnitude spectrum of `block`, in dB, for the first
// `n_bins` bins. `block` must be WATERFALL_NFFT samples.
//
// Scaled by 1/NFFT and floored with a small epsilon, so a silent input
// gives a very negative number rather than negative infinity.
std::vector<double> spectrum_db(const std::vector<double>& block, int n_bins);

// Map a spectrum onto exactly `width` columns.
//
// **Peak-hold, not point-sampling, when shrinking.** The carriers are
// one or two bins wide and about six bins apart, so taking every k'th
// bin drops some of them outright and leaves a ragged comb where the
// signal should be a solid block -- and it looks like a reception
// problem rather than a display one. Taking the maximum over each
// output column's bins keeps every carrier visible.
//
// Widening interpolates instead, to avoid a blocky frequency axis on a
// wide pane.
std::vector<double> reduce_to_width(const std::vector<double>& values, int width);

}  // namespace sstvae::dsp

#endif
