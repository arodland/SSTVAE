#include "dsp/spectrum.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <numbers>

#include "dsp/fft.hpp"

namespace sstvae::dsp {

namespace {

// numpy's `hanning`, which is the symmetric window (denominator N-1),
// not the periodic one. The display would survive either, but matching
// the reference means a spectrum can be compared against it directly
// when something looks wrong.
const std::vector<double>& hann_window() {
    static const std::vector<double> window = [] {
        std::vector<double> w(WATERFALL_NFFT);
        for (int i = 0; i < WATERFALL_NFFT; ++i) {
            w[i] = 0.5 - 0.5 * std::cos(2.0 * std::numbers::pi * i /
                                        (WATERFALL_NFFT - 1));
        }
        return w;
    }();
    return window;
}

}  // namespace

std::vector<double> spectrum_db(const std::vector<double>& block, int n_bins) {
    if (static_cast<int>(block.size()) < WATERFALL_NFFT || n_bins <= 0) return {};
    const std::vector<double>& window = hann_window();

    std::vector<cdouble> buffer(WATERFALL_NFFT);
    for (int i = 0; i < WATERFALL_NFFT; ++i) buffer[i] = block[i] * window[i];
    const std::vector<cdouble> spectrum = fft(buffer, true);

    const int limit = std::min(n_bins, WATERFALL_NFFT / 2 + 1);
    std::vector<double> out(limit);
    for (int i = 0; i < limit; ++i) {
        out[i] = 20.0 * std::log10(std::abs(spectrum[i]) / WATERFALL_NFFT + 1e-12);
    }
    return out;
}

std::vector<double> reduce_to_width(const std::vector<double>& values, int width) {
    const int n = static_cast<int>(values.size());
    if (width <= 0 || n == 0) return {};
    if (width == n) return values;

    std::vector<double> out(width);
    if (width > n) {
        // Linear interpolation across the source, endpoints included --
        // np.interp over linspace(0, n-1, width).
        for (int i = 0; i < width; ++i) {
            const double at = width == 1 ? 0.0
                                         : static_cast<double>(i) * (n - 1) /
                                               (width - 1);
            const int lo = static_cast<int>(at);
            const int hi = std::min(lo + 1, n - 1);
            out[i] = values[lo] + (at - lo) * (values[hi] - values[lo]);
        }
        return out;
    }

    // Group boundaries are strictly increasing because n >= width, so
    // no output column is empty -- which is what lets the max below be
    // taken over a non-empty range without a special case.
    for (int i = 0; i < width; ++i) {
        const int start = static_cast<int>(static_cast<long long>(i) * n / width);
        const int stop = i + 1 == width
                             ? n
                             : static_cast<int>(
                                   static_cast<long long>(i + 1) * n / width);
        out[i] = *std::max_element(values.begin() + start, values.begin() + stop);
    }
    return out;
}

}  // namespace sstvae::dsp
