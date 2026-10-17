// Waveform L, the receive side the receiver uses: L's exact symbol matched
// filter and PULSE_NORM. Port of sstvae/qrss/lwave.py's matched_filter and
// PULSE_NORM (the reference; lwave.hpp holds the transmit side, and
// to_data_centre, which fell out of it). The loopback diagnostics
// (loopback_stats and the masks behind it) are not ported: nothing in the
// receiver calls them.
//
//     m_p = (G/L_T_NUM) sum_n z[n] p_L((t_n - pos_p T_L)/T_L),  z centred on the data
//
// ce_rx::matched_filter with L's pulse table (lwave::pulse_table(), 970
// grid steps a symbol, 7760 either side), so bitwise as that one is.

#ifndef SSTVAE_QRSS_LWAVE_RX_HPP
#define SSTVAE_QRSS_LWAVE_RX_HPP

#include <complex>
#include <cstdint>
#include <span>
#include <vector>

#include "qrss/rx_config.hpp"

namespace sstvae::qrss::lwave_rx {

using cplx = std::complex<double>;

// PULSE_NORM = sqrt(sum_m rrc(m/970)^2 / 970), m = -7760..7760: the value
// lwave.py computes at import (rx_config.hpp, generated from it).
inline constexpr double PULSE_NORM = rxconfig::lwave::PULSE_NORM;

// The same number computed here from ce::rrc and numpy's pairwise sum, as
// the reference computes it (the ctest holds it equal to PULSE_NORM).
double pulse_norm();

// matched_filter(z, fs, n0, pos): m_p for each position, samples outside z
// counting 0; a constant 1 gives lwave::g0(). fs must divide 16000
// ("fs must divide 16000, not {fs}"); an empty z with any position is
// std::out_of_range, numpy's IndexError.
std::vector<double> matched_filter(std::span<const double> z, std::int64_t fs, std::int64_t n0,
                                   std::span<const std::int64_t> pos);
std::vector<cplx> matched_filter(std::span<const cplx> z, std::int64_t fs, std::int64_t n0,
                                 std::span<const std::int64_t> pos);

}  // namespace sstvae::qrss::lwave_rx

#endif  // SSTVAE_QRSS_LWAVE_RX_HPP
