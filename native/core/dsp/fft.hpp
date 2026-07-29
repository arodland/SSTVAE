// FFT, via the vendored pocketfft header.
//
// Wrapped rather than used directly so the rest of the core sees one
// small interface, and so swapping the backend later touches one file.
// See native/third_party/pocketfft/README.md for why FFT-derived values
// are tolerance-class against the Python reference rather than bitwise
// (SciPy's backend is ducc0, pocketfft's successor -- shared lineage,
// no guarantee of identical bits, and an FFT could not be bitwise
// across platforms anyway).

#pragma once

#include <complex>
#include <cstddef>
#include <vector>

#include "pocketfft_hdronly.h"

namespace sstvae::dsp {

using cdouble = std::complex<double>;

// In-place-free complex FFT. `forward` selects the sign convention;
// the inverse is scaled by 1/n so that ifft(fft(x)) == x, matching
// numpy and scipy.
inline std::vector<cdouble> fft(const std::vector<cdouble>& x, bool forward) {
    const std::size_t n = x.size();
    std::vector<cdouble> out(n);
    if (n == 0) return out;
    const pocketfft::shape_t shape{n};
    const pocketfft::stride_t stride{
        static_cast<std::ptrdiff_t>(sizeof(cdouble))};
    const double scale = forward ? 1.0 : 1.0 / static_cast<double>(n);
    pocketfft::c2c(shape, stride, stride, pocketfft::shape_t{0}, forward,
                   x.data(), out.data(), scale);
    return out;
}

}  // namespace sstvae::dsp
