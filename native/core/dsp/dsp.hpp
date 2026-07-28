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

// Envelope clip-and-filter for PAPR (PEP) control.
//
// SSB transmitters are limited by envelope peak power, so clipping acts
// on the analytic-signal magnitude, not raw samples. Iterated because
// the bandpass regrows peaks after each clip. Returns unit-RMS.
std::vector<double> tx_condition(std::span<const double> x,
                                 double clip_headroom_db, int iterations = 2);

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
