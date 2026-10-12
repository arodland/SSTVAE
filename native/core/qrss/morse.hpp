// Morse keying of QRSSTVAE's callsign windows (spec 2.7, design 2.7).
// Port of sstvae/qrss/morse.py (the reference).
//
// A format module. Each callsign window is CW_UNITS = 192 Morse units;
// units 0..7 are plain carrier, the call starts at unit 8 and has
// CW_ROOM_UNITS = 176 units of room, and the rest is plain carrier.
// Standard timing: dot 1 unit, dash 3, 1 unit between elements, 3
// between characters, no trailing gap.
//
// The alphabet is config::MORSE (A-Z, 0-9, '/') and nothing else: a
// header callsign must be sendable (decision D10), so an unsendable
// character is refused, never substituted. This is deliberately not
// core/dsp/morse, the SSTVAE CW ID, which is lenient about its input.
//
// Callsigns are UTF-8; refusals count and quote code points exactly as
// Python does (qrss/pytext.hpp), so their text matches the reference.

#ifndef SSTVAE_QRSS_MORSE_HPP
#define SSTVAE_QRSS_MORSE_HPP

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace sstvae::qrss::morse {

// The code of `ch` (".-" for 'A'), or an empty view if it has none.
std::string_view code_of(char32_t ch);

// The callsign with trailing spaces stripped, or std::invalid_argument
// with morse.check_callsign's text: 1 to MAX_CALL_CHARS characters, all
// in the table. Lowercase is refused, not folded.
std::string check_callsign(std::string_view callsign);

// Key-down (1) / key-up (0) per unit for the call itself.
std::vector<std::uint8_t> morse_keying(std::string_view callsign);

// CW_UNITS values: the call in units CW_LEAD_UNITS .. CW_LEAD_UNITS+len-1.
// std::invalid_argument if it does not fit CW_ROOM_UNITS (no valid call
// fails: the longest, "00000000", takes 173).
std::vector<std::uint8_t> keying_units(std::string_view callsign);

}  // namespace sstvae::qrss::morse

#endif  // SSTVAE_QRSS_MORSE_HPP
