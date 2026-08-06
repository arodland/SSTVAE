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

// Cache a handful of FFT plans (twiddle factors) instead of rebuilding
// one on every call. Must be defined before pocketfft_hdronly.h is
// first included, which happens right below -- this is that header's
// only include site in the codebase.
//
// Off (0) by default upstream. A live-decode-loop perf capture showed
// 5.45% of total CPU time in `cfftp<double>::cfftp` -- the plan
// *constructor*, not the transform itself -- because every FFT call
// anywhere in the native code, including 67 back-to-back same-length
// calls inside a single acquire_blind() invocation, was rebuilding its
// twiddle-factor table from scratch. The cache is a thread-safe,
// mutex-protected LRU keyed by transform length (pocketfft_hdronly.h's
// own `get_plan`), so this only removes redundant setup work; the
// transform math, and so every golden vector's tolerance, is unchanged.
// Sized generously above the small, fixed set of distinct lengths this
// codebase's hot paths actually use, not tuned to it exactly -- eviction
// churn from an undersized cache would be worse than the memory a few
// extra unused slots cost.
#ifndef POCKETFFT_CACHE_SIZE
#define POCKETFFT_CACHE_SIZE 16
#endif

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
