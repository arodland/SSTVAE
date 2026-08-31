// Small DSP helpers shared by the modem.
//
// Port of sstvae/modem/dsp.py. The FIR design here replicates
// scipy.signal.firwin exactly rather than approximating it: the filters
// are part of the waveform (the transmit bandpass shapes what goes on
// air) and part of acquisition (the sync lowpass sets what the preamble
// detector sees), so "a reasonable windowed sinc" is not good enough.

#pragma once

#include <complex>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "config.hpp"

namespace sstvae::dsp {

using cdouble = std::complex<double>;

// Real passband -> complex baseband (pure heterodyne by FCENTER).
//
// Deliberately unfiltered: any FIR long enough to be selective smears
// beyond the 32-sample cyclic prefix and causes ISI, while the
// 160-sample demod correlation already nulls the -f-FCENTER heterodyne
// image exactly (all image spacings are carrier-spacing multiples) and
// provides per-carrier noise selectivity. Sync filters its own copy.
std::vector<cdouble> to_baseband(std::span<const double> x);

// to_baseband(x), but as if `x` were a slice of a longer signal starting
// at absolute sample index `start_sample` rather than at 0 -- i.e.
// exactly to_baseband(full)[start_sample : start_sample + x.size()] for
// whatever longer `full` array `x` actually came from. See
// sstvae.modem.dsp.to_baseband_at's docstring: needed by any caller
// that baseband-converts one long recording as a series of independent
// chunks and carries state *across* them (sync::BlindAccumulator's live
// caller in rx/engine.cpp), because to_baseband always treats its own
// input's first sample as the heterodyne's local n=0.
std::vector<cdouble> to_baseband_at(std::span<const double> x, std::int64_t start_sample);

// Fractional part of a phase in cycles, in [0, 1). For arbitrary
// frequencies the product cannot be made exact the way the integer
// cases can, but reducing before exp() removes the large-argument error.
double wrap_cycles(double cycles);

std::vector<cdouble> freq_correct(std::span<const cdouble> z, double f_hz);

// scipy.signal.firwin with the Hamming window and scale=True, which is
// what the reference calls. `bands` is a single passband: {0, cutoff}
// for a lowpass, {lo, hi} for a bandpass. Frequencies in Hz.
std::vector<double> firwin_lowpass(int numtaps, double cutoff_hz);
std::vector<double> firwin_bandpass(int numtaps, double lo_hz, double hi_hz);

// The modified Bessel function of the first kind, order 0 -- `np.i0`,
// which is what scipy's Kaiser window is defined in terms of.
double bessel_i0(double x);

// scipy.signal.windows.kaiser(M, beta, sym=True).
std::vector<double> kaiser(int m, double beta);

// scipy.signal.resample_poly(x, up, down) with its default
// window=('kaiser', 5.0) and padtype='constant'.
//
// **Rate conversion is stateful in the live capture path** -- see
// `audio::StreamResampler`, and the note in CLAUDE.md about the 4.7 dB
// that per-chunk resampling cost. This function is the whole-signal
// form, correct for a file or a complete waveform and wrong for a
// stream of blocks.
//
// The Kaiser-windowed sinc filter this designs depends only on (up,
// down), not on `x`, so `filter_cache`, if given, is a slot this fills
// in on first use and reuses on every later call with it -- for a
// caller (StreamResampler) that resamples many chunks at one fixed
// ratio and would otherwise redesign the same (up to ~8821-tap) filter
// from scratch on every chunk. Owned entirely by the caller: no shared
// state, no locking, no eviction policy -- the caller's own lifetime is
// the cache's lifetime. Passing the same slot across two different
// (up, down) ratios is a caller bug, same as reusing any other
// single-purpose local for two things, and is not detected.
std::vector<double> resample_poly(std::span<const double> x, int up, int down,
                                  std::optional<std::vector<double>>* filter_cache = nullptr);

// numpy.convolve(a, v, mode="same"): the centre len(a) samples of the
// full convolution, for len(a) >= len(v).
std::vector<double> convolve_same(std::span<const double> a,
                                  std::span<const double> v);
std::vector<cdouble> convolve_same(std::span<const cdouble> a,
                                   std::span<const double> v);

// Selective lowpass used only for preamble detection, where FIR
// smearing is harmless and out-of-band noise would degrade the
// autocorrelation metric.
std::vector<cdouble> sync_lowpass(std::span<const cdouble> z);

// scipy.signal.hilbert: the analytic signal, via FFT.
std::vector<cdouble> hilbert(std::span<const double> x);

// scipy.signal.fftconvolve(a, v, mode="valid"), for len(a) >= len(v).
//
// FFT-based rather than a direct sum, matching the reference's choice.
// That is not just convention: a direct moving sum and an FFT
// convolution differ in the last bits, and `sync` feeds these straight
// into argmax over a metric, so keeping the two implementations in the
// same numerical family keeps the disagreement at FFT-vs-FFT level
// instead of algorithm-vs-algorithm.
//
// Real and complex overloads mirror scipy's own split: it uses a real
// FFT (and `next_fast_len(..., real=True)`) for real inputs, and the
// transform length affects the rounding, so the choice is copied rather
// than unified.
std::vector<cdouble> fftconvolve_valid(std::span<const cdouble> a,
                                       std::span<const cdouble> v);
std::vector<double> fftconvolve_valid(std::span<const double> a,
                                      std::span<const double> v);

// scipy.fft.next_fast_len. pocketfft's good_size_* is the same function
// scipy exposes -- verified equal over a range of sizes, which matters
// because blind acquisition derives its CFO bin spacing from the
// transform length.
std::size_t next_fast_len(std::size_t n, bool real = false);

// Envelope clip-and-filter for PAPR (PEP) control.
//
// SSB transmitters are limited by envelope peak power, so clipping acts
// on the analytic-signal magnitude, not raw samples. Iterated because
// the bandpass regrows peaks after each clip. Returns unit-RMS.
//
// `overshoot` is one factor per pass -- its length *is* the pass count,
// so nothing can disagree about how many passes ran. A factor above 1
// clips past the threshold so the next filter regrows back toward it
// rather than above it (CESSB's "more than clipping"), applied as
// scale**k. See sstvae/config.py's CLIP_OVERSHOOT for the measurements.
std::vector<double> tx_condition(
    std::span<const double> x, double clip_headroom_db,
    std::span<const double> overshoot = config::CLIP_OVERSHOOT);

// Envelope (PEP) peak-to-average power ratio in dB.
double papr_db(std::span<const double> x);

// Scale to `peak` of full scale and round to int16.
//
// Rounding is half-to-even, matching np.round -- std::round would go
// half-away-from-zero and differ on exact .5 values, which are not rare
// after a peak normalization.
std::vector<std::int16_t> to_int16(std::span<const double> x,
                                   double peak = 0.95);

}  // namespace sstvae::dsp
