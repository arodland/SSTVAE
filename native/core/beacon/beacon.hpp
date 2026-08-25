// Beacon side-channel: a continuously repeating, self-describing packet
// carried as BPSK chips on the one reserved carrier
// (config::BEACON_CARRIER), CHIPS_PER_FRAME chips per frame, for the
// entire transmission.
//
// Port of sstvae/modem/beacon.py. Each repetition ("superframe") is:
//
//     sync word (Barker-13, unmodulated marker)
//     | Golay(24,12)-coded payload (7 codewords = 84 payload bits)
//
// Payload = 10-bit absolute frame counter (index of the frame whose data
// symbols carry the sync word's first chip) + 48-bit callsign
// (8 chars x 6 bits) + 2-bit mode index + 8 reserved bits (transmitted
// as config::BEACON_RESERVED_VALUE, ignored on decode) + 16-bit CRC over
// everything before it. Exactly 84 bits -- the same 7 Golay chunks the
// pre-mode-field format padded out to (PROTOCOL_VERSION 4; see
// sstvae/modem/beacon.py for the full rationale).
//
// Because the counter is absolute (not modulo the superframe period), a
// receiver needs no prior knowledge of where the transmission started:
// any window containing one full, correctly-decoded superframe reveals
// exactly which frame index it landed on.

#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "config.hpp"

namespace sstvae::beacon {

inline constexpr int SYNC_LEN = config::BEACON_SYNC_LEN;

// 10 + 48 + 2 + 8 + 16 = 84 payload bits: exactly a whole number of
// 12-bit Golay chunks, no padding.
inline constexpr int PAYLOAD_BITS = config::BEACON_COUNTER_BITS +
                                    config::BEACON_CALLSIGN_BITS +
                                    config::BEACON_MODE_BITS +
                                    config::BEACON_RESERVED_BITS +
                                    config::BEACON_CRC_BITS;
inline constexpr int N_CHUNKS = (PAYLOAD_BITS + 11) / 12;         // 7
inline constexpr int PADDED_PAYLOAD_BITS = N_CHUNKS * 12;          // 84
static_assert(PAYLOAD_BITS == PADDED_PAYLOAD_BITS,
              "the v4 payload fills its Golay chunks exactly");
inline constexpr int CODED_LEN = N_CHUNKS * 24;                    // 168
inline constexpr int SUPERFRAME_LEN = SYNC_LEN + CODED_LEN;        // 181
inline constexpr int MAX_FRAME_COUNTER = (1 << config::BEACON_COUNTER_BITS) - 1;
inline constexpr int MAX_MODE_INDEX = (1 << config::BEACON_MODE_BITS) - 1;

// A window needs at least 2*SUPERFRAME_LEN-1 chips to *guarantee* a
// full, uncut superframe regardless of phase. Below this, decode() may
// still succeed if the phase is lucky, but is not guaranteed to.
inline constexpr int MIN_FRAMES_FOR_SYNC =
    (2 * SUPERFRAME_LEN - 1 + config::CHIPS_PER_FRAME - 1) / config::CHIPS_PER_FRAME;

// 64-symbol alphabet for 6-bit callsign characters. Amateur callsigns
// are uppercase letters/digits/slash; the rest of the code space is
// filled with harmless punctuation so any 6-bit value round-trips to a
// printable character.
//
// Duplicated from beacon.py rather than generated, because it is 64
// bytes and a generated artifact would cost more than it saves -- but
// the golden corpus pins every code point, so a divergence is caught
// rather than assumed away.
inline constexpr std::string_view ALPHABET =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/-. ?!@#$%^&*()_+=~[]{}<>:;,";
static_assert(ALPHABET.size() == 64);

struct BeaconResult {
    std::int64_t chip_offset;  // index into the input where sync starts
    int frame_index;           // absolute frame index (from the counter)
    std::string callsign;
    // Transmitted mode index, raw. A value with no config::MODES entry
    // (3 today) is a mode this receiver does not know; consumers fall
    // back to assume-mode-C rather than rejecting the packet.
    int mode_index;
};

// String -> BEACON_CALLSIGN_CHARS 6-bit codes, space-padded/truncated.
std::vector<int> callsign_to_codes(std::string_view callsign);
std::string codes_to_callsign(std::span<const int> codes);

// CRC-16/CCITT-FALSE over a 0/1 bit array, MSB-first, init 0xFFFF.
std::vector<int> crc16(std::span<const int> bits);

// One superframe repetition -> SUPERFRAME_LEN chips in {-1, +1}.
std::vector<double> encode_chips(int frame_index, std::string_view callsign,
                                 int mode_index);

// Continuous beacon chip stream, CHIPS_PER_FRAME chips per frame,
// covering frames [start_frame, start_frame + n_frames): superframes
// back to back (truncating the last if it does not fit), each labelled
// with the absolute frame index its sync word lands on.
std::vector<double> chip_stream(int start_frame, int n_frames,
                                std::string_view callsign, int mode_index);

// Offsets where the Barker-13 sync word plausibly starts, best
// correlation first, normalized so results are comparable across signal
// levels.
std::vector<std::int64_t> find_sync(std::span<const double> chips,
                                    double threshold = 0.6,
                                    int max_candidates = 8);

// Decode exactly one repetition's SUPERFRAME_LEN-SYNC_LEN=CODED_LEN
// coded chips (i.e. `chips[off+SYNC_LEN : off+SYNC_LEN+CODED_LEN]` for
// some sync offset `off`), or nullopt if its CRC does not check out.
// `decode()` tries this at every find_sync() candidate before falling
// back to combining evidence across repetitions (see the .cpp); exposed
// separately so a caller (or a test) that wants single-repetition
// behaviour specifically doesn't have to reconstruct it.
std::optional<BeaconResult> decode_single_repetition(
    std::int64_t chip_offset, std::span<const double> coded_chips);

// Find and decode one beacon superframe anywhere in `chips` (soft
// values, any length >= SUPERFRAME_LEN). Tries the best-correlated sync
// candidates in order and returns the first one whose CRC checks out;
// falls back to combining evidence across every repetition in `chips`
// if no single one decodes alone -- see the .cpp for why and how (port
// of sstvae/modem/beacon.py's module docstring on multi-repetition
// combining).
std::optional<BeaconResult> decode(std::span<const double> chips,
                                   double threshold = 0.6);

}  // namespace sstvae::beacon
