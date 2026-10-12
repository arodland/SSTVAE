#include "qrss/morse.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>

#include "qrss/config.hpp"
#include "qrss/pytext.hpp"

namespace sstvae::qrss::morse {

std::string_view code_of(char32_t ch) {
    for (const config::MorseCode& m : config::MORSE)
        if (static_cast<char32_t>(static_cast<unsigned char>(m.ch)) == ch) return m.code;
    return {};
}

std::string check_callsign(std::string_view callsign) {
    // callsign.rstrip(" "): ASCII spaces only, and 0x20 never occurs
    // inside a multi-byte UTF-8 sequence, so stripping bytes is the same.
    std::string_view call = callsign;
    while (!call.empty() && call.back() == ' ') call.remove_suffix(1);
    const std::u32string cps = pytext::decode(call);
    if (cps.size() < 1 || cps.size() > static_cast<std::size_t>(config::MAX_CALL_CHARS))
        throw std::invalid_argument("callsign " + pytext::repr(callsign) + " must be 1 to " +
                                    std::to_string(config::MAX_CALL_CHARS) + " characters");
    // sorted({c for c in call if c not in MORSE}), printed as a list.
    std::u32string bad;
    for (char32_t c : cps)
        if (code_of(c).empty()) bad += c;
    if (!bad.empty()) {
        std::sort(bad.begin(), bad.end());
        bad.erase(std::unique(bad.begin(), bad.end()), bad.end());
        std::string list = "[";
        for (std::size_t i = 0; i < bad.size(); ++i) {
            if (i) list += ", ";
            list += pytext::repr(std::u32string_view(&bad[i], 1));
        }
        list += "]";
        throw std::invalid_argument("callsign " + pytext::repr(callsign) +
                                    " has characters outside [A-Z0-9/]: " + list);
    }
    return std::string(call);
}

std::vector<std::uint8_t> morse_keying(std::string_view callsign) {
    const std::string call = check_callsign(callsign);
    std::vector<std::uint8_t> out;
    for (std::size_t i = 0; i < call.size(); ++i) {   // ASCII by now
        if (i) out.insert(out.end(), 3, 0);
        const std::string_view code = code_of(static_cast<unsigned char>(call[i]));
        for (std::size_t j = 0; j < code.size(); ++j) {
            if (j) out.push_back(0);
            out.insert(out.end(), code[j] == '.' ? 1 : 3, 1);
        }
    }
    return out;
}

std::vector<std::uint8_t> keying_units(std::string_view callsign) {
    const std::vector<std::uint8_t> k = morse_keying(callsign);
    if (k.size() > static_cast<std::size_t>(config::CW_ROOM_UNITS))
        throw std::invalid_argument("callsign " + pytext::repr(callsign) + " needs " +
                                    std::to_string(k.size()) + " Morse units, more than the window's " +
                                    std::to_string(config::CW_ROOM_UNITS));
    std::vector<std::uint8_t> out(config::CW_UNITS, 0);
    std::copy(k.begin(), k.end(), out.begin() + config::CW_LEAD_UNITS);
    return out;
}

}  // namespace sstvae::qrss::morse
