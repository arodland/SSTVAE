// Waveform CE, the receive side the receiver uses: the mean waveform (the
// tracker's and EM's template) and the exact symbol matched filter. Port
// of sstvae/qrss/ce.py's mean_template, _mean_template_grid and
// matched_filter (the reference; ce.hpp holds the transmit side they are
// built on). symbol_classes and the loopback diagnostics are not ported:
// nothing in the receiver calls them.
//
//     E[s(t)] = keyed * a * exp(j(phi_mu + 2 pi theta_cw)) * exp(-v/2)
//     m_p = (L/T_NUM) sum_n z[n] p((t_n - pos_p T)/T),  t_n = t0 + (n0 + n)/fs
//
// with mu, nu per stream symbol (known symbols mu +-1 nu 0, unknown data mu
// 0 nu 1, EM's soft data mu x_hat nu v).
//
// How close to Python, and why:
//
// - matched_filter is **bitwise**: the per-position row of products is
//   numpy's (a real tap times a complex sample through numpy's fused
//   complex multiply, lwave::np_mul) summed as np.sum(axis=1) sums a row
//   (npsum's pairwise sum onto 0), then scaled by L / T_NUM -- so it does
//   not depend on `chunk` either.
// - mean_template is ce.phase_at / phase_grid, var_grid, cw_phase_turns
//   and _turns_to_phasor (ce.hpp: bitwise where libm agrees with numpy,
//   as on x86-64 Linux) multiplied in the reference's order, with one
//   exception: np.exp(-v/2) is numpy's own SIMD exp, 1 ulp from libm's on
//   a few percent of arguments, so the template holds to ~1e-16 relative
//   rather than bitwise wherever v != 0.
//
// Refusals are ce.hpp's (std::invalid_argument with the reference's
// ValueError texts: the sym/var shapes, fs, a keying), plus numpy's
// IndexError text for a matched filter over an empty z.

#ifndef SSTVAE_QRSS_CE_RX_HPP
#define SSTVAE_QRSS_CE_RX_HPP

#include <complex>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "qrss/ce.hpp"
#include "qrss/frame.hpp"

namespace sstvae::qrss::ce_rx {

using cplx = std::complex<double>;
using frame::FrameSpec;

// mean_template(tau_pos, mu, nu, spec, keying, ook, keyed_from_pos): the
// mean waveform at each tau (symbols from t0). mu and nu hold n_sym values
// ("sym has shape ..." / "var has shape ..." otherwise); `keying` a valid
// keying (ce::window_keying) or null for None; the keyed span starts at
// `keyed_from_pos` (nullopt: spec.keyed_start_pos) and has no ramp.
std::vector<cplx> mean_template(std::span<const double> tau, std::span<const double> mu, std::span<const double> nu,
                                const FrameSpec& spec, const ce::Keying* keying = nullptr, bool ook = false,
                                std::optional<double> keyed_from_pos = std::nullopt);

// _mean_template_grid(fs, n0, n, mu, nu, spec, keying, ook): the same at
// t0 + (n0 + i)/fs, i < n, through the exact grid path (fs divides 16000).
std::vector<cplx> mean_template_grid(std::int64_t fs, std::int64_t n0, std::int64_t n, std::span<const double> mu,
                                     std::span<const double> nu, const FrameSpec& spec,
                                     const ce::Keying* keying = nullptr, bool ook = false);

// matched_filter(z, fs, n0, pos): m_p for each position p, samples outside
// z counting 0. A constant 1 gives G0. fs must divide 16000; an empty z
// with any position is std::out_of_range ("index -1 is out of bounds for
// axis 0 with size 0", as numpy's clip-then-index).
std::vector<double> matched_filter(std::span<const double> z, std::int64_t fs, std::int64_t n0,
                                   std::span<const std::int64_t> pos);
std::vector<cplx> matched_filter(std::span<const cplx> z, std::int64_t fs, std::int64_t n0,
                                 std::span<const std::int64_t> pos);

// The matched filter for any pulse table on the 16 kHz grid: `t_num` grid
// steps a symbol, `half` steps either side (table has 2 half + 1 values),
// at grid step `step` (16000 / fs). CE's above and lwave_rx's are this
// with their own constants.
std::vector<double> matched_filter_table(std::span<const double> z, std::int64_t step, std::int64_t n0,
                                         std::span<const std::int64_t> pos, std::int64_t t_num, std::int64_t half,
                                         std::span<const double> table);
std::vector<cplx> matched_filter_table(std::span<const cplx> z, std::int64_t step, std::int64_t n0,
                                       std::span<const std::int64_t> pos, std::int64_t t_num, std::int64_t half,
                                       std::span<const double> table);

}  // namespace sstvae::qrss::ce_rx

#endif  // SSTVAE_QRSS_CE_RX_HPP
