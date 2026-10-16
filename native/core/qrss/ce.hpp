// Waveform CE, the transmit side: the phase pulse, phase synthesis on the
// exact 16 kHz grid and in closed form, the callsign-window phase, the
// keying envelope, baseband and 8 kHz audio. Port of sstvae/qrss/ce.py
// (the reference), whose module docstring is the definition:
//
//     s(t) = a(t) * exp(j * [phi(t) + 2 pi theta_cw(t)])
//     phi(t) = beta * sum_s x_s * p((t - t0)/T - pos(s))
//
// with time in symbols from t0 throughout (tau = (t - t0)/T).
//
// A format module: what this computes goes on the air. No RNG, and every
// constant comes from the generated qrss/qrss_config.hpp and qrss/tables.hpp
// (or is derived from them here); the few that ce.py defines itself are
// marked below and compared with it by tests/test_qrss_native_ce.py.
//
// How close to Python, and why:
//
// - **Bitwise**: the 16 kHz grid sum (`grid_sum`, `phase_grid`, `var_grid`).
//   The reference runs scipy's upfirdn over the frozen pulse table; its
//   loop adds, for each output sample, the products of the positions
//   that reach it in ascending position order onto 0.0, and the products
//   it adds besides (zero taps, zero positions) are signed zeros that
//   cannot move a nonzero sum. So the direct 17-term sum here, in the
//   same order, is the same number -- and chunk-invariant.
// - **Bitwise where libm agrees with numpy** (it does on x86-64 Linux,
//   where numpy's float64 sin/cos/complex exp call glibc): `rrc`,
//   `pulse`, `phase_at`, `hann_int`, `cw_phase_turns`, `ramp_envelope`,
//   `turns_to_phasor`, `baseband` and `to_audio`, each written in the
//   reference's operation order. to_audio's carrier multiply also copies
//   numpy's SIMD complex multiply, which fuses: Re(z e) =
//   fma(zr, er, -(zi * ei)). Elsewhere these hold to 1e-12 (rad, or
//   relative to unit amplitude), the plan's bar.
// - **Exact integers**: the carrier phase of `to_audio`, n * num mod
//   (den * 8000) by a portable mulmod -- the reference's numpy int64
//   product is the same number wherever it does not overflow (it needs
//   n0 near 1e9 samples with den > 380,000 to).
//
// Refusals are std::invalid_argument with the reference's ValueError text.
// `check_carrier` and `keyed_span_pos` quote the caller's value; from C++
// it is shown as Python's repr of the double (the binding shows the
// caller's own object, as Python's f-string does).

#ifndef SSTVAE_QRSS_CE_HPP
#define SSTVAE_QRSS_CE_HPP

#include <array>
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

namespace sstvae::qrss::ce {

using frame::FrameSpec;

// The 16 kHz grid the pulse table lives on, and half the table: 8 symbols.
inline constexpr std::int64_t GRID_FS = config::T_DEN;
inline constexpr std::int64_t HALF = std::int64_t(config::SPAN) * config::T_NUM;   // 3880
inline constexpr std::size_t N_OFFSETS = 2 * config::SPAN + 1;                   // j = -8..8

// ce.py's own constants (not in constants.py, so not generated): on-off
// keying follows the Morse in units 5..183 of a window (design 2.7, D14),
// and FSK keys one cycle per unit below the carrier.
inline constexpr int OOK_FIRST_KEYED_UNIT = config::CE_OOK_FIRST_KEYED_UNIT;
inline constexpr int OOK_LAST_KEYED_UNIT = config::CE_OOK_LAST_KEYED_UNIT;
inline constexpr double FSK_SHIFT_HZ = -1.0 / (config::CW_UNIT * config::T_SYM);   // -16.495 Hz

// float(ALPHA), float(BETA) and BETA**2 as the reference computes them.
inline constexpr double ALPHA_F = static_cast<double>(config::ALPHA.num) / static_cast<double>(config::ALPHA.den);
inline constexpr double BETA_F = static_cast<double>(config::BETA.num) / static_cast<double>(config::BETA.den);

// --- pulse --------------------------------------------------------------------

// _rrc: the textbook unit-energy RRC h(tau), uncut and unscaled, with its
// two removable singularities (tau = 0 and 4 alpha |tau| = 1) as the
// reference evaluates them.
double rrc(double tau);

// PULSE_NORM: sqrt(sum_m h(m/485)^2 / 485) over m = -3880..3880 (numpy's
// pairwise sum), computed once from `rrc`.
double pulse_norm();

// pulse: p(tau) = h(tau) / PULSE_NORM for |tau| <= 8, else 0.
double pulse(double tau);

// PULSE_TABLE: p(m/485), m = -3880..3880, the frozen table (tables.hpp).
std::span<const double> pulse_table();

// G0 = PULSE_TABLE.sum() / 485, the matched-filter gain of a constant.
double g0();

// _grid_step: 16 kHz grid steps per sample at rate fs. Refuses an fs that
// does not divide 16000: "fs must divide 16000, not {fs}".
std::int64_t grid_step(std::int64_t fs);

// --- symbols on the position axis ---------------------------------------------

// _check_sym: "{name} has shape {shape}, expected ({n_sym},)" unless the
// array's shape is (n_sym,).
void check_sym(const FrameSpec& spec, std::span<const std::int64_t> shape, std::string_view name = "sym");

// at_positions: float64[n_pos], stream-order values at their positions,
// 0 in the windows. `sym` must hold n_sym values.
std::vector<double> at_positions(std::span<const double> sym, const FrameSpec& spec);

// --- phase ---------------------------------------------------------------------

// _pulse_offsets for one d in [-1/2, 1/2]: p(d - j) for j = -8..8 (index
// j + 8), or its square, by the reference's shared sin/cos expansion.
std::array<double, N_OFFSETS> pulse_offsets(double d, bool square);

// phase_at: phi = beta sum_s sym_s p(tau - pos_s) at each tau (symbols from
// t0), in closed form; with `var` also v = beta^2 sum_s var_s p(...)^2.
// sym (and var) hold n_sym values.
std::vector<double> phase_at(std::span<const double> tau, std::span<const double> sym, const FrameSpec& spec);
std::pair<std::vector<double>, std::vector<double>> phase_at(std::span<const double> tau,
                                                             std::span<const double> sym,
                                                             std::span<const double> var,
                                                             const FrameSpec& spec);

// _grid_sum: out[i] = sum_p vals[p] * table[L*(n0 + i) - 485 p + 3880],
// over the positions p in [0, len(vals)) the reference's window reaches,
// in ascending p onto 0.0 (bitwise the reference's upfirdn). n <= 0 gives
// an empty vector; fs must divide 16000.
std::vector<double> grid_sum(std::int64_t fs, std::int64_t n0, std::int64_t n, std::span<const double> vals,
                             std::span<const double> table);

// phase_grid: phi at t0 + (n0 + i)/fs, i < n, on the exact table path.
std::vector<double> phase_grid(std::int64_t fs, std::int64_t n0, std::int64_t n, std::span<const double> sym,
                               const FrameSpec& spec);

// var_grid: beta^2 sum_s var_s p^2 on the same grid.
std::vector<double> var_grid(std::int64_t fs, std::int64_t n0, std::int64_t n, std::span<const double> var,
                             const FrameSpec& spec);

// --- callsign windows -----------------------------------------------------------

// _hann_int: R(y), the integral of the unit-width Hann CDF centred on 0.
double hann_int(double y);

// A frame's callsign keying: n_win rows of CW_UNITS 0/1 values, row-major
// (what window_keying returns; None is the absence of one).
using Keying = std::vector<std::uint8_t>;

// window_keying from one mask (shape (192,), every window keyed alike) or
// one per window (shape (n_win, 192)): `values` in C order, `shape` as the
// array has it. Refuses any other shape, or a value other than 0 and 1:
// "keying must be 0/1 of shape (192,) or (n_win, 192), not {shape}" (a
// 1-D shape is reported broadcast to (n_win, len), as the reference does).
Keying window_keying(std::span<const double> values, std::span<const std::int64_t> shape, const FrameSpec& spec);
// That refusal's text for an array of `shape`.
std::string keying_refusal(const FrameSpec& spec, std::span<const std::int64_t> shape);
// The same for a 0/1 mask of 192 units (morse.keying_units), every window
// keyed alike.
Keying window_keying(std::span<const std::uint8_t> units, const FrameSpec& spec);

// window_units: x = (tau - (P_w - 1/2)) / 2, time in Morse units from the
// start of window w. A negative w counts from the end, as Python indexes a
// tuple; anything else outside the windows is std::out_of_range
// ("tuple index out of range").
double window_units(double tau, const FrameSpec& spec, std::int64_t w);

// cw_phase_turns: (theta in turns, a) at each tau. FSK: theta follows the
// keying with Hann-integral transitions and is the integer -(key-down units
// so far) after each window; a = 1. OOK: theta = 0 and a = 0 in units
// 5..183 of a window except where keyed. `keying` null is None (no call:
// plain carrier for FSK, carrier off in units 5..183 for OOK); otherwise it
// must already be a valid (n_win, 192) keying.
struct CwPhase {
    std::vector<double> theta;
    std::vector<double> a;
};
CwPhase cw_phase_turns(std::span<const double> tau, const FrameSpec& spec, const Keying* keying, bool ook = false);

// --- keyed span ---------------------------------------------------------------------

// A lead-in outside [0, LEAD_IN_MAX_S] (NaN included). The text shows the
// value as Python's repr of the double; the binding rewords it with the
// caller's own object, which is what the reference's f-string shows.
struct LeadInRefusal : std::invalid_argument {
    using std::invalid_argument::invalid_argument;
};
// "lead-in must be 0 to 10 s, not {shown}"
std::string lead_in_refusal(std::string_view shown);

// keyed_span_pos: [start, end) of the keyed carrier in symbols from t0, the
// start moved lead_in_s earlier. LeadInRefusal for a bad lead-in.
std::pair<double, double> keyed_span_pos(const FrameSpec& spec, double lead_in_s = 0.0);

// _ramp_envelope: 1 on [lo, hi), raised-cosine ramps of ramp_sym outside
// it, 0 beyond.
double ramp_envelope(double tau, double lo, double hi, double ramp_sym);
std::vector<double> ramp_envelope(std::span<const double> tau, double lo, double hi, double ramp_sym);

// _turns_to_phasor: exp(j 2 pi turns), the turns reduced to [0, 1) first.
std::complex<double> turns_to_phasor(double turns);

// --- baseband and audio ---------------------------------------------------------------

struct BasebandOptions {
    const Keying* keying = nullptr;   // a valid keying (window_keying), or none
    bool ook = false;
    double lead_in_s = 0.0;
    double ramp_s = 0.05;
    double time_scale = 1.0;
    std::int64_t chunk = std::int64_t(1) << 18;
};

// What baseband settles before it synthesizes anything, in the reference's
// order: n0 = round(start_s * fs) (half to even; NaN is
// std::invalid_argument and inf std::overflow_error, Python's round()
// texts), whether the exact table path applies (time_scale 1, an integer
// fs dividing 16000 -- 0 is ZeroDivision, as 16000 % 0 -- and start_s
// within 1e-9 of that rate's sample grid), and the keyed span [lo, hi)
// (LeadInRefusal).
//
// A round(start_s * fs) past int64 is no refusal here: the reference's
// Python int only fails where the exact path builds its int64 sample
// numbers, so `n0_overflows` defers that to the first chunk baseband
// synthesizes (std::overflow_error, numpy's text), and the closed form,
// which never uses n0, is unaffected.
struct BasebandPlan {
    std::int64_t n0 = 0;
    bool exact = false;
    bool n0_overflows = false;   // round(start_s * fs) is outside int64 (n0 is then 0)
    double lo = 0.0, hi = 0.0;
};
BasebandPlan baseband_plan(const FrameSpec& spec, double fs, double start_s, const BasebandOptions& opt = {});

// baseband: complex128[n], the CE signal at t0 + (start_s + i/fs) *
// time_scale: a * exp(j(phi + 2 pi theta_cw)), zero where unkeyed, with a
// raised-cosine ramp of ramp_s just outside the keyed span. The exact
// path (baseband_plan) evaluates phi on the 16 kHz grid; anything else in
// closed form (phase_at). sym holds n_sym values. Computed in chunks of
// `chunk` samples (>= 1); the result does not depend on it.
std::vector<std::complex<double>> baseband(std::span<const double> sym, const FrameSpec& spec, double fs,
                                           double start_s, std::int64_t n, const BasebandOptions& opt = {});

// to_audio: float64 8 kHz audio, amplitude * Re{z e^{j 2 pi f n / 8000}}
// with f = num/den Hz and n counted from n0. z is resampled to 8 kHz
// first (scipy's resample_poly) when fs is not 8000. The carrier phase is
// the exact integer turn (n * num mod (den * 8000)) / (den * 8000).
// "carrier_hz_den must be positive" unless den > 0.
std::vector<double> to_audio(std::span<const std::complex<double>> z, std::int64_t fs, std::int64_t num,
                             std::int64_t den = 1, double amplitude = 0.5, std::int64_t n0 = 0);

// (a * b) mod m for a, b >= 0 and m > 0 (to_audio passes a, b < m), with no
// 128-bit type.
std::int64_t mulmod(std::int64_t a, std::int64_t b, std::int64_t m);

// check_carrier: the carrier in Hz, or "carrier {shown} Hz is outside the
// 300-2700 Hz audio band that receivers search" (NaN and inf included).
// `shown` is how the caller's value prints; the first form shows the
// double as Python's repr does.
std::string carrier_refusal(std::string_view shown);
double check_carrier(double carrier_hz);
double check_carrier(double carrier_hz, std::string_view shown);

// carrier_fraction: (num, den) of a carrier given as a float or a str
// (Fraction(x).limit_denominator(1 << 20)), an integer, or a Rational
// (taken as it is, as the reference takes a Fraction).
inline constexpr std::int64_t CARRIER_MAX_DEN = std::int64_t(1) << 20;
Rational carrier_fraction(double carrier_hz);
Rational carrier_fraction(std::string_view carrier_hz);
Rational carrier_fraction(std::int64_t carrier_hz);
inline Rational carrier_fraction(const Rational& carrier_hz) { return carrier_hz; }

}  // namespace sstvae::qrss::ce

#endif  // SSTVAE_QRSS_CE_HPP
