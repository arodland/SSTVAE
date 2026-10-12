// GENERATED FILE -- DO NOT EDIT.
// Written by tools/gen_qrss_config_header.py from sstvae/qrss.
// Regenerate with: python tools/gen_qrss_config_header.py
//
// Every QRSSTVAE format constant, for the C++ port (native/core/qrss/).
// SSTVAE's own numbers (TRANSMIT_LATENTS_PER_GROUP and friends) are in
// ../config.hpp, generated from sstvae/config.py, and are not restated.

#pragma once

#include <array>
#include <cstdint>
#include <string_view>

namespace sstvae::qrss::config {

// An exact rational, as fractions.Fraction is in Python.
struct Ratio {
    std::int64_t num;
    std::int64_t den;
    constexpr double value() const { return static_cast<double>(num) / static_cast<double>(den); }
};

// --- scalars (sstvae/qrss/constants.py) ---
inline constexpr int FS = 8000;
inline constexpr int T_NUM = 485;
inline constexpr int T_DEN = 16000;
inline constexpr int SPAN = 8;
inline constexpr int N_PRE = 660;
inline constexpr int N_HDR = 2640;
inline constexpr int REF_PERIOD = 16;
inline constexpr int N_HDR_BITS = 2474;
inline constexpr int N_INFO_BITS = 142;
inline constexpr int POLAR_N = 2048;
inline constexpr int FORMAT_VERSION = 2;
inline constexpr int WAVEFORM_CE = 0;
inline constexpr int WAVEFORM_L = 1;
inline constexpr int LEAD_IN_MAX_S = 10;
inline constexpr int CW_WINDOW = 384;
inline constexpr int CW_UNIT = 2;
inline constexpr int CW_UNITS = 192;
inline constexpr int CW_LEAD_UNITS = 8;
inline constexpr int CW_ROOM_UNITS = 176;
inline constexpr int INT8_SCALE = 20;
inline constexpr int FE_FS = 4000;
inline constexpr int FE_CENTER_HZ = 1500;
inline constexpr int CH_FS = 250;
inline constexpr int CH_DECIM = 16;
inline constexpr int L_T_NUM = 970;
inline constexpr int L_N_PRE = 330;
inline constexpr int L_N_HDR = 1320;
inline constexpr int L_CW_WINDOW = 192;
inline constexpr double T_SYM = 0x1.f0a3d70a3d70ap-6;  // 0.0303125
inline constexpr double A0 = 0x1.73c9ce7c81bcep-1;  // 0.7261490370736909
inline constexpr double K_LIN = 0x1.296e3eca0163fp-1;  // 0.5809192296589528
inline constexpr double CARRIER_FRAC = 0x1.0df945c9630cep-1;  // 0.5272924240430485
inline constexpr double USEFUL_FRAC = 0x1.5990fd2abc397p-2;  // 0.33746715138755107
inline constexpr double SPREAD_RHO = 0x1.ba5e353f7ced9p-5;  // 0.054
inline constexpr double Z_ACCEPT = 0x1.8000000000000p+2;  // 6.0
inline constexpr double L_T_SYM = 0x1.f0a3d70a3d70ap-5;  // 0.060625
inline constexpr double L_CLIP_HEADROOM_DB = 0.0;  // 0.0
inline constexpr double L_CLIP_GUARD_HZ = 0x1.0000000000000p+0;  // 1.0
inline constexpr Ratio ALPHA{3, 20};  // 0.15
inline constexpr Ratio BETA{4, 5};  // 0.8
inline constexpr Ratio L_EDGE_HZ{920, 97};  // 9.484536082474227
inline constexpr Ratio L_PILOT_HZ{12207, 970};  // 12.584536082474227
inline constexpr Ratio L_PILOT_FRAC{3, 20};  // 0.15
inline constexpr std::array<int, 2> FORMAT_VERSIONS{1, 2};
inline constexpr std::array<int, 4> CW_AFTER_FULL{13888, 28350, 42812, 57274};
inline constexpr std::array<int, 4> L_CW_AFTER_FULL{6944, 14175, 21406, 28637};
inline constexpr std::array<double, 2> CARRIER_BAND_HZ{0x1.2c00000000000p+8, 0x1.5180000000000p+11};  // (300.0, 2700.0)
inline constexpr std::array<double, 3> L_CLIP_OVERSHOOT{0x1.0000000000000p+0, 0x1.8000000000000p+0, 0x1.0000000000000p+1};  // (1.0, 1.5, 2.0)

// --- SHA-256 domain strings (sstvae/qrss/sequences.py) ---
inline constexpr std::string_view PREAMBLE_DOMAIN = "QRSSTVAE CE preamble";
inline constexpr std::string_view REFERENCE_DOMAIN = "QRSSTVAE CE reference";
inline constexpr std::string_view SCRAMBLE_DOMAIN = "QRSSTVAE scramble";
inline constexpr std::string_view SPREAD_DOMAIN = "QRSSTVAE CE spread header";
inline constexpr std::string_view L_PREAMBLE_DOMAIN = "QRSSTVAE L preamble";
inline constexpr std::string_view L_REFERENCE_DOMAIN = "QRSSTVAE L reference";

// The Python module's own consistency asserts.
static_assert(T_DEN % FS == 0 && FE_FS == CH_FS * CH_DECIM, "rates nest");
static_assert(CW_LEAD_UNITS + CW_ROOM_UNITS + CW_LEAD_UNITS == CW_UNITS);
static_assert(L_PILOT_HZ.num == 12207 && L_PILOT_HZ.den == 970 && 2 * L_CW_WINDOW == CW_WINDOW);

// --- frame presets (frame.PRESETS, LEGACY, PROTOTYPES; lframe.PRESETS) ---
// The parameters only; lengths are derived in frame.cpp exactly as
// frame.FrameSpec / lframe.LFrameSpec derive them. `hdr_rho` is the
// resolved value (FrameSpec.__post_init__ has already applied the default).
enum class PresetGroup { Presets, Legacy, Prototypes, LPresets };
struct PresetDef {
    std::string_view name;
    PresetGroup group;
    int waveform;
    int n_pre;
    int n_hdr;
    int n_data;
    int n_win;
    std::array<int, 4> cw_after;
    double hdr_rho;
};
inline constexpr std::array<PresetDef, 12> FRAME_PRESETS{{
    {"full", PresetGroup::Presets, 0, 660, 2640, 50600, 4, {13888, 28350, 42812, 57274}, 0x1.ba5e353f7ced9p-5},
    {"medium", PresetGroup::Presets, 0, 660, 2640, 16384, 2, {10000, 20777, 0, 0}, 0x1.ba5e353f7ced9p-5},
    {"short", PresetGroup::Presets, 0, 660, 2640, 4096, 2, {5000, 7670, 0, 0}, 0x1.ba5e353f7ced9p-5},
    {"tiny", PresetGroup::Presets, 0, 660, 0, 1024, 0, {0, 0, 0, 0}, 0.0},
    {"full-v1", PresetGroup::Legacy, 0, 660, 2640, 50600, 4, {13888, 28350, 42812, 57274}, 0.0},
    {"medium-v1", PresetGroup::Legacy, 0, 660, 2640, 16384, 2, {10000, 20777, 0, 0}, 0.0},
    {"short-v1", PresetGroup::Legacy, 0, 660, 2640, 4096, 2, {5000, 7670, 0, 0}, 0.0},
    {"full-spreadonly", PresetGroup::Prototypes, 0, 660, 0, 50600, 4, {11248, 25710, 40172, 54634}, 0x1.ba5e353f7ced9p-5},
    {"l-full", PresetGroup::LPresets, 1, 330, 1320, 50600, 4, {6944, 14175, 21406, 28637}, 0x1.ba5e353f7ced9p-5},
    {"l-medium", PresetGroup::LPresets, 1, 330, 1320, 16384, 2, {5000, 10388, 0, 0}, 0x1.ba5e353f7ced9p-5},
    {"l-short", PresetGroup::LPresets, 1, 330, 1320, 4096, 2, {2500, 3834, 0, 0}, 0x1.ba5e353f7ced9p-5},
    {"l-tiny", PresetGroup::LPresets, 1, 330, 0, 1024, 0, {0, 0, 0, 0}, 0.0},
}};

// --- polar information set (sstvae/qrss/polar.py), ascending ---
inline constexpr std::string_view POLAR_INFO_SET_SHA256 = "4e064eee9b1df68ee4d044970b51f05a16700d51599f857f86e7d15d782b97bb";
inline constexpr std::array<int, 142> POLAR_INFO_SET{{
    1015, 1019, 1021, 1022, 1023, 1503, 1519, 1527, 1530, 1531, 1532, 1533,
    1534, 1535, 1663, 1727, 1759, 1771, 1773, 1774, 1775, 1779, 1781, 1782,
    1783, 1785, 1786, 1787, 1788, 1789, 1790, 1791, 1853, 1854, 1855, 1879,
    1883, 1885, 1886, 1887, 1895, 1899, 1901, 1902, 1903, 1907, 1909, 1910,
    1911, 1912, 1913, 1914, 1915, 1916, 1917, 1918, 1919, 1935, 1943, 1947,
    1949, 1950, 1951, 1959, 1961, 1962, 1963, 1964, 1965, 1966, 1967, 1969,
    1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981,
    1982, 1983, 1989, 1990, 1991, 1993, 1994, 1995, 1996, 1997, 1998, 1999,
    2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012,
    2013, 2014, 2015, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025,
    2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035, 2036, 2037,
    2038, 2039, 2040, 2041, 2042, 2043, 2044, 2045, 2046, 2047,
}};

// --- Morse (sstvae/qrss/morse.py), in the table's own order ---
struct MorseCode {
    char ch;
    std::string_view code;
};
inline constexpr int MAX_CALL_CHARS = 8;
inline constexpr std::array<MorseCode, 37> MORSE{{
    {'A', ".-"},
    {'B', "-..."},
    {'C', "-.-."},
    {'D', "-.."},
    {'E', "."},
    {'F', "..-."},
    {'G', "--."},
    {'H', "...."},
    {'I', ".."},
    {'J', ".---"},
    {'K', "-.-"},
    {'L', ".-.."},
    {'M', "--"},
    {'N', "-."},
    {'O', "---"},
    {'P', ".--."},
    {'Q', "--.-"},
    {'R', ".-."},
    {'S', "..."},
    {'T', "-"},
    {'U', "..-"},
    {'V', "...-"},
    {'W', ".--"},
    {'X', "-..-"},
    {'Y', "-.--"},
    {'Z', "--.."},
    {'0', "-----"},
    {'1', ".----"},
    {'2', "..---"},
    {'3', "...--"},
    {'4', "....-"},
    {'5', "....."},
    {'6', "-...."},
    {'7', "--..."},
    {'8', "---.."},
    {'9', "----."},
    {'/', "-..-."},
}};

}  // namespace sstvae::qrss::config
