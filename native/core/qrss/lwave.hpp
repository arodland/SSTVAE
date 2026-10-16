// Waveform L, the transmit side: the data on a sample grid, the pilot tone
// with its keying, callsign windows and lead-in, the composite baseband,
// SSTVAE's clipper, the clipped frame at 250 Hz and its band-limited
// interpolation to 8 kHz. Port of sstvae/qrss/lwave.py (the reference).
//
// A format module: this is what goes on the air for waveform L. In
// baseband centred on the pilot (the carrier a sender names),
//
//     z0 = d(t) e^{-j 2 pi f_p (t - t0)} + pilot(t)
//
// and the transmitted signal is z0 through three clipper passes, each
// pass's correction filtered to the occupied band, then scaled by 1/A_id
// (lwave.py's module docstring has the whole definition).
//
// How close to Python, and why (the plan's parity classes):
//
// - Bit for bit: at_positions, mix_turns (exact integer turns modulo the
//   rational's grid, den * fs), tau_grid, keyed_span_pos, window_keying,
//   data_grid and _grid_sum (scipy's upfirdn, filters::upfirdn), and
//   resample given the same frame -- its sums run in scipy's order, one
//   output sample at a time, so the result is also independent of how a
//   caller chunks the output (chunk-invariant bitwise).
// - Bit for bit where libm is numpy's: _raised, cw_turns, id_weight, pilot,
//   composite and band_mask take sin/cos from libm (numpy's float64 sin and
//   cos call libm on x86-64 glibc), multiply complex numbers as numpy's
//   SIMD loop does (the real part fused: fma(ar, br, -(ai*bi)), see
//   np_mul), and copy numpy's operation order everywhere else. Against
//   another libm the bar is 1e-12.
// - Tolerance: clip and frame_baseband. The filter is a whole-frame FFT
//   (pocketfft here, scipy.fft's ducc0 in Python: the same algorithms, not
//   the same bits) and the 1.5 overshoot goes through pow (libm here,
//   numpy's SVML pow on AVX-512 machines); about a third of the samples
//   cross the threshold on the first pass. Measured: about 1e-15 per FFT,
//   a few 1e-15 on the clipped frame; the bar is 1e-9.
//
// The magnitude is numpy's complex abs (larger * sqrt(fma(r, r, 1)) with
// r = smaller/larger), not hypot, and the final 1/A_id is numpy's complex
// division by a real scalar (a multiply by 1/A_id).
//
// Errors are std::invalid_argument carrying the reference's ValueError
// text. at_positions, data_grid, composite and frame_baseband need an L
// frame (Python fails with an AttributeError on another waveform's, here
// frame::WrongWaveform); the pilot's functions take any frame, as the
// reference's do.

#ifndef SSTVAE_QRSS_LWAVE_HPP
#define SSTVAE_QRSS_LWAVE_HPP

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "qrss/qrss_config.hpp"
#include "qrss/frame.hpp"
#include "qrss/rational.hpp"

namespace sstvae::qrss::lwave {

using cplx = std::complex<double>;

// --- constants ----------------------------------------------------------------

inline constexpr int GRID_FS = config::T_DEN;                                   // the pulse table's 16 kHz grid
inline constexpr std::int64_t HALF = std::int64_t(config::SPAN) * config::L_T_NUM;   // 7760 grid steps
inline constexpr int CLIP_FS = config::CH_FS;                                   // 250 Hz: the clipper's rate

// lwave.py's own module constants, which tools/gen_qrss_config_header.py
// does not emit yet (see the port's shared_changes_needed); the parity test
// holds each against the reference's value.
inline constexpr double L_AVG_POWER = config::L_AVG_POWER;     // the clipped data's average power, before 1/A_id
inline constexpr int ID_RAMP_UNITS = config::L_ID_RAMP_UNITS;         // pilot level ramps over units 0..7 and 184..191
inline constexpr double KEY_RAMP_S = config::L_KEY_RAMP_S;      // keying ramps outside the keyed span
inline constexpr int INTERP_HALF = config::L_INTERP_HALF;         // interpolator taps either side at 8 kHz
inline constexpr double MARGIN_S = config::L_MARGIN_S;         // frame_baseband's default margin

// Derived exactly as the reference derives them (sqrt and pow, both
// correctly rounded or libm's, and numpy's pairwise sum for G0).
double a_pilot();   // sqrt(3/17): the pilot's amplitude, symbols at unit power
double a_clip();    // sqrt(1 + A_PILOT^2) * 10^(L_CLIP_HEADROOM_DB/20) = sqrt(20/17)
double a_id();      // sqrt(L_AVG_POWER): the pilot in the lead-in and the windows
double g0();        // PULSE_TABLE.sum() / 970: the matched filter's gain on a constant

// lwave.PULSE_TABLE: p(m/970), m = -7760..7760 (the frozen table).
std::span<const double> pulse_table();

// --- numpy's arithmetic ---------------------------------------------------------

// numpy's complex128 multiply a*b. Its SIMD loop (AVX2, AVX-512, NEON) fuses
// the real part's first product and the imaginary part's first product:
// re = fma(ar, br, -(ai*bi)), im = fma(ar, bi, ai*br). A real operand is
// numpy's cast, (r, +0).
inline cplx np_mul(cplx a, cplx b) {
    return {std::fma(a.real(), b.real(), -(a.imag() * b.imag())),
            std::fma(a.real(), b.imag(), a.imag() * b.real())};
}
inline cplx np_mul(double a, cplx b) { return np_mul(cplx(a, 0.0), b); }

// numpy's complex128 absolute: larger * sqrt(fma(r, r, 1)), r = smaller/larger
// (0 when larger is 0), with inf and nan handled as numpy does.
double np_abs(cplx z);

// --- the data on a sample grid -------------------------------------------------------

// ce._grid_step: 16 kHz grid steps per sample at rate fs; "fs must divide
// 16000, not {fs}" otherwise.
std::int64_t grid_step(std::int64_t fs);

// _grid_sum: sum_p vals[p] * table[G*(n0+i) - 970 p + 7760] for i < n, by
// scipy's upfirdn (filters::upfirdn), complex or real. n <= 0 gives none.
std::vector<cplx> grid_sum(std::int64_t fs, std::int64_t n0, std::int64_t n, std::span<const cplx> vals,
                           std::span<const double> table);
std::vector<double> grid_sum(std::int64_t fs, std::int64_t n0, std::int64_t n, std::span<const double> vals,
                             std::span<const double> table);

// "sym has shape S, expected (n_sym,)" unless `shape` is (n_sym,).
void check_sym_shape(const frame::FrameSpec& spec, std::span<const std::int64_t> shape);

// at_positions: complex128[n_pos], the stream-order symbols at their
// positions, 0 in the windows. `sym` is (n_sym,).
std::vector<cplx> at_positions(std::span<const cplx> sym, const frame::FrameSpec& spec);

// data_grid: d(t) at t = t0 + (n0 + i)/fs, centred on the data.
std::vector<cplx> data_grid(std::int64_t fs, std::int64_t n0, std::int64_t n, std::span<const cplx> sym,
                            const frame::FrameSpec& spec);

// mix_turns: f (t - t0) in turns, reduced to [0, 1), at t = t0 + (n0 + i)/fs:
// ((n mod P) * (num mod P) mod P) / P with P = den * fs, in exact integers
// (numpy's int64 products wrap once P exceeds about 3e9; these do not).
// fs must be positive; a P past int64 is std::overflow_error, as numpy
// refuses it.
std::vector<double> mix_turns(std::int64_t fs, std::int64_t n0, std::int64_t n, const Rational& f_hz);

// tau_grid: (t - t0)/T at the samples, from the exact grid index.
std::vector<double> tau_grid(std::int64_t fs, std::int64_t n0, std::int64_t n);

// --- pilot: keying, callsign windows and lead-in -------------------------------------

// A lead-in outside [0, LEAD_IN_MAX_S]: "lead-in must be 0 to 10 s, not
// {lead_in_s}", the value shown as Python's float repr (a binding replaces
// it by str() of what it was given).
struct LeadInRefusal : std::invalid_argument {
    explicit LeadInRefusal(double lead);
    double lead;
};
std::string lead_in_refusal(std::string_view shown);

// keyed_span_pos: [start, end) of keying in L symbols from t0. `end` is
// spec.keyed_end_pos() (an integer value unless the last window ends the frame).
std::pair<double, double> keyed_span_pos(const frame::FrameSpec& spec, double lead_in_s = 0.0);

// Callsign keying as the reference receives it: np.asarray(keying)'s shape
// and its values as float64 (C order). Only 0 and 1 are accepted; a value
// that is not a real number is NaN here, which is refused as Python refuses it.
struct Keying {
    std::vector<std::int64_t> shape;
    std::vector<double> values;
    // One (192,) mask of 0/1 units, as morse.keying_units returns it.
    static Keying units(std::span<const std::uint8_t> mask);
};

// window_keying: (n_win, 192) uint8 in C order, from none (nullopt), one
// (192,) mask (broadcast to every window) or one mask per window.
std::optional<std::vector<std::uint8_t>> window_keying(const Keying* keying, const frame::FrameSpec& spec);

// _raised: 0 for y <= 0, 1 for y >= 1, a raised cosine between.
double raised(double y);

// cw_turns: theta in turns at tau (L symbols from t0), the callsign FSK of
// the pilot (ce.cw_phase_turns's FSK with one unit = one L symbol).
std::vector<double> cw_turns(std::span<const double> tau, const frame::FrameSpec& spec, const Keying* keying);

// id_weight: W(tau), 1 where the pilot is at average power (the lead-in and
// window units 8..183), raised-cosine ramps around them.
std::vector<double> id_weight(std::span<const double> tau, const frame::FrameSpec& spec, double lead_in_s = 0.0);

// pilot: the pilot at tau, before the clipper:
// env * (A_PILOT (1 - W) + A_ID W e^{j 2 pi theta}).
std::vector<cplx> pilot(std::span<const double> tau, const frame::FrameSpec& spec, const Keying* keying = nullptr,
                        double lead_in_s = 0.0, double ramp_s = KEY_RAMP_S);

// --- composite and clipper ------------------------------------------------------------

// composite: z0 at t0 + (n0 + i)/fs in pilot-centred baseband (data mixed down
// by f_p, plus the pilot). n <= 0 gives none.
std::vector<cplx> composite(std::span<const cplx> sym, const frame::FrameSpec& spec, std::int64_t fs, std::int64_t n0,
                            std::int64_t n, const Keying* keying = nullptr, double lead_in_s = 0.0,
                            double ramp_s = KEY_RAMP_S);

// band_mask: the clipper's filter, [-f_p - EDGE - guard, +guard] in
// pilot-centred Hz with raised-cosine edges `taper_hz` wide.
std::vector<double> band_mask(std::span<const double> freqs, double guard_hz = config::L_CLIP_GUARD_HZ,
                              double taper_hz = 1.0);

// scipy.fft.next_fast_len(target) for a complex transform: the smallest
// 11-smooth number >= target (target itself up to 12).
std::int64_t next_fast_len(std::int64_t target);

// numpy.fft.fftfreq(n, d): k / (n*d) for k = 0..(n-1)//2, then -(n//2)..-1.
std::vector<double> fftfreq(std::int64_t n, double d);

// clip: SSTVAE's clipper on z (pilot-centred, rate fs): per pass k,
// z += F{(min(threshold/|z|, 1)^k - 1) z}, F the band mask on an FFT padded
// to next_fast_len(len + 4 fs). Exponent 1 is a copy and 2 a square, as in
// numpy's scalar power; anything else (1.5) goes through pow.
std::vector<cplx> clip(std::span<const cplx> z, std::int64_t fs = CLIP_FS, double threshold = a_clip(),
                       std::span<const double> overshoot = config::L_CLIP_OVERSHOOT,
                       double guard_hz = config::L_CLIP_GUARD_HZ);

// The clipped frame at `fs`, pilot-centred, scaled to unit average power:
// z[i] is at t0 + (n0 + i)/fs and is zero outside the keyed span and its ramps.
struct Baseband {
    std::vector<cplx> z;
    std::int64_t n0 = 0;
    std::int64_t fs = CLIP_FS;
};

// frame_baseband: the whole frame through the clipper. n0 is
// floor((t_lo - margin_s) fs) and the frame runs to ceil((t_hi + margin_s) fs),
// as in the reference (the clipper's FFT padding depends on it).
Baseband frame_baseband(std::span<const cplx> sym, const frame::FrameSpec& spec, const Keying* keying = nullptr,
                        double lead_in_s = 0.0, std::int64_t fs = CLIP_FS, double ramp_s = KEY_RAMP_S,
                        double margin_s = MARGIN_S);

// --- to 8 kHz ---------------------------------------------------------------------

// _interp_taps(up): the frozen lowpass (firwin(481, 125, fs=8000,
// ('kaiser', 12)), tables::L_INTERP_TAPS) times `up`.
std::vector<double> interp_taps(std::int64_t up);

// resample: the frame z (first sample at n0, rate fs) at t0 + (i0 + i)/fs_out,
// i < out.size(), by band-limited interpolation; fs_out a multiple of fs
// ("fs_out must be a multiple of {fs}" otherwise). Each output sample is its
// own sum, in scipy's upfirdn order, so any chunking gives the same bits.
void resample_into(std::span<const cplx> z, std::int64_t n0, std::int64_t fs, std::int64_t fs_out, std::int64_t i0,
                   std::span<cplx> out);
std::vector<cplx> resample(std::span<const cplx> z, std::int64_t n0, std::int64_t fs, std::int64_t fs_out,
                           std::int64_t i0, std::int64_t m);
std::vector<cplx> resample(const Baseband& bb, std::int64_t fs_out, std::int64_t i0, std::int64_t m);

// to_data_centre: z mixed up by f_p, so the data sit at 0 Hz (receive side;
// it falls out of mix_turns).
std::vector<cplx> to_data_centre(std::span<const cplx> z, std::int64_t n0, std::int64_t fs);

}  // namespace sstvae::qrss::lwave

#endif  // SSTVAE_QRSS_LWAVE_HPP
