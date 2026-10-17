// GENERATED FILE -- DO NOT EDIT.
// Written by tools/gen_qrss_config_header.py from sstvae/qrss.
// Regenerate with: python tools/gen_qrss_config_header.py
//
// Frozen float tables of the QRSSTVAE format. Defined in tables.cpp.

#pragma once

#include <array>
#include <string_view>

namespace sstvae::qrss::tables {

// ce.PULSE_TABLE: p(m/485), m = -3880..3880
// sha256 of the little-endian float64 bytes: 7f69e973f1374d2d0b184b4b1798179e4c85068bbafbd50e8c85c234d0b0b96f
extern const std::array<double, 7761> CE_PULSE_TABLE;
inline constexpr std::string_view CE_PULSE_TABLE_SHA256 = "7f69e973f1374d2d0b184b4b1798179e4c85068bbafbd50e8c85c234d0b0b96f";

// lwave.PULSE_TABLE: p(m/970), m = -7760..7760
// sha256 of the little-endian float64 bytes: 94b6e691d735eab5f00d8fd9372f440dc35eec6bf22aca4e20c120965aff840b
extern const std::array<double, 15521> L_PULSE_TABLE;
inline constexpr std::string_view L_PULSE_TABLE_SHA256 = "94b6e691d735eab5f00d8fd9372f440dc35eec6bf22aca4e20c120965aff840b";

// lwave interpolator, unit gain (multiply by `up`)
// sha256 of the little-endian float64 bytes: 3bfac46f5b0db8be68530ff728b3e4c8f8f1255067a91b159cad13da86b4104f
extern const std::array<double, 481> L_INTERP_TAPS;
inline constexpr std::string_view L_INTERP_TAPS_SHA256 = "3bfac46f5b0db8be68530ff728b3e4c8f8f1255067a91b159cad13da86b4104f";

}  // namespace sstvae::qrss::tables
