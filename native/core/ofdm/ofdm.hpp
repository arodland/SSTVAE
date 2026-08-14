// DFT-matrix OFDM: complex carrier amplitudes <-> waveform samples.
//
// Port of sstvae/modem/ofdm.py. Passband modulation generates the real
// transmit waveform directly. Demodulation operates on the complex
// baseband signal produced by dsp::to_baseband, where carrier k sits at
// bin (k - 11) * RS Hz.

#pragma once

#include <array>
#include <cstdint>
#include <complex>
#include <span>
#include <vector>

#include "config.hpp"

namespace sstvae::ofdm {

using cdouble = std::complex<double>;
using config::M;
using config::NC;
using config::NSYM;

// Passband carrier frequencies, Hz, and their baseband images.
const std::array<double, NC>& carrier_freqs();
const std::array<double, NC>& baseband_freqs();

// (NSYM, NC) passband modulation matrix, row-major. Symbol samples
// n = 0..NSYM-1 with the phase reference at the start of the useful part
// (n = NCP). Carriers are multiples of RS, so the first NCP samples are
// a true cyclic prefix.
std::span<const cdouble> mod_matrix();

// (NC, M) baseband demodulation matrix over one useful window, row-major.
std::span<const cdouble> demod_matrix();

// The fixed unit-magnitude Zadoff-Chu sequence used for the preamble and
// the frame pilots. Built from config::PILOT_PHASE_NUM, which is
// generated from the Python reference -- see the note in config.hpp.
const std::array<cdouble, NC>& pilot_sequence();

// (n_sym, NC) complex symbols, row-major -> real waveform of
// n_sym * NSYM samples.
std::vector<double> modulate_symbols(std::span<const cdouble> symbols,
                                     std::size_t n_sym);

// Demodulate one useful window of baseband signal starting at `start`
// (nominal index of the first useful sample). `backoff` shifts the
// window earlier into the cyclic prefix; the resulting linear phase
// slope is absorbed by pilot equalization as long as it is applied
// consistently. Factor 2 undoes the amplitude halving of the
// real->analytic conversion.
//
// A window running off the end of the signal is zero-padded, matching
// the Python reference, because that edge decides what happens at the
// tail of a recording.
//
// Unlike the reference this *throws* when start - backoff is negative.
// Python reaches that case through a negative numpy slice, which wraps
// to the end of the array and returns confident garbage; there is no
// behaviour there worth reproducing, and sync never produces it.
std::array<cdouble, NC> demod_window(std::span<const cdouble> z, std::int64_t start,
                                     std::int64_t backoff = 0);

// Real passband preamble: the pilot symbol, periodic with M over the
// whole block (double-length CP + two periods).
std::vector<double> preamble_waveform();

// Complex baseband replica of the preamble, for timing correlation.
std::vector<cdouble> preamble_template();

// Complex baseband replica of one bare frame-pilot symbol's useful
// window (no CP) -- used by blind acquisition to find frame timing from
// the pilot's own per-frame periodicity, without the (non-repeating)
// transmission-start preamble.
std::vector<cdouble> pilot_template();

}  // namespace sstvae::ofdm
