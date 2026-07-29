#include "beacon/beacon.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <utility>

#include "golay/golay.hpp"

namespace sstvae::beacon {
namespace {

using config::BEACON_CALLSIGN_CHAR_BITS;
using config::BEACON_CALLSIGN_CHARS;
using config::BEACON_COUNTER_BITS;
using config::BEACON_CRC_BITS;
using config::CHIPS_PER_FRAME;

void append_int_bits(std::vector<int>& out, int value, int width) {
    for (int i = width - 1; i >= 0; --i) out.push_back((value >> i) & 1);
}

int bits_to_int(std::span<const int> bits) {
    int v = 0;
    for (int b : bits) v = (v << 1) | (b & 1);
    return v;
}

int code_for_char(char c) {
    const std::size_t pos = ALPHABET.find(c);
    // Anything outside the alphabet becomes a space, as the reference's
    // dict .get(c, code_of_space) does.
    return pos == std::string_view::npos
               ? static_cast<int>(ALPHABET.find(' '))
               : static_cast<int>(pos);
}

std::vector<int> payload_bits(int frame_index, std::string_view callsign) {
    if (frame_index < 0 || frame_index > MAX_FRAME_COUNTER)
        throw std::invalid_argument(
            "beacon: frame_index exceeds the counter range");
    std::vector<int> body;
    append_int_bits(body, frame_index, BEACON_COUNTER_BITS);
    for (int code : callsign_to_codes(callsign))
        append_int_bits(body, code, BEACON_CALLSIGN_CHAR_BITS);

    std::vector<int> out = body;
    const std::vector<int> crc = crc16(body);
    out.insert(out.end(), crc.begin(), crc.end());
    return out;
}

}  // namespace

std::vector<int> callsign_to_codes(std::string_view callsign) {
    std::string s;
    for (char c : callsign.substr(0, std::min<std::size_t>(
                                        callsign.size(), BEACON_CALLSIGN_CHARS)))
        s.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(c))));
    s.resize(BEACON_CALLSIGN_CHARS, ' ');
    std::vector<int> out;
    out.reserve(BEACON_CALLSIGN_CHARS);
    for (char c : s) out.push_back(code_for_char(c));
    return out;
}

std::string codes_to_callsign(std::span<const int> codes) {
    std::string out;
    for (int c : codes) out.push_back(ALPHABET[static_cast<std::size_t>(c & 0x3F)]);
    // rstrip(): the reference strips trailing whitespace only.
    while (!out.empty() && std::isspace(static_cast<unsigned char>(out.back())))
        out.pop_back();
    return out;
}

std::vector<int> crc16(std::span<const int> bits) {
    // CRC-16/CCITT-FALSE, written the way the reference writes it: the
    // data bit is shifted in at the LSB and the polynomial applied when
    // the *pre-shift* top bit was set. Algebraically the standard
    // algorithm; transcribed rather than rederived so the two cannot
    // disagree on an edge case.
    std::uint32_t reg = 0xFFFF;
    for (int b : bits) {
        const std::uint32_t top = (reg >> 15) & 1u;
        reg = ((reg << 1) & 0xFFFFu) | static_cast<std::uint32_t>(b & 1);
        if (top) reg ^= 0x1021u;
    }
    std::vector<int> out;
    append_int_bits(out, static_cast<int>(reg), BEACON_CRC_BITS);
    return out;
}

std::vector<double> encode_chips(int frame_index, std::string_view callsign) {
    std::vector<int> padded = payload_bits(frame_index, callsign);
    padded.resize(PADDED_PAYLOAD_BITS, 0);

    std::vector<double> out;
    out.reserve(SUPERFRAME_LEN);
    for (int i = 0; i < SYNC_LEN; ++i)
        out.push_back(static_cast<double>(config::BEACON_SYNC[static_cast<std::size_t>(i)]));
    for (int i = 0; i < PADDED_PAYLOAD_BITS; i += 12) {
        const int data = bits_to_int(std::span<const int>(padded).subspan(
            static_cast<std::size_t>(i), 12));
        for (int bit : golay::codeword_bits(data))
            out.push_back(1.0 - 2.0 * static_cast<double>(bit));
    }
    return out;
}

std::vector<double> chip_stream(int start_frame, int n_frames,
                                std::string_view callsign) {
    const std::size_t n_chips =
        static_cast<std::size_t>(n_frames) * CHIPS_PER_FRAME;
    std::vector<double> out(n_chips);
    std::size_t pos = 0;
    while (pos < n_chips) {
        const int frame_index =
            start_frame + static_cast<int>(pos / CHIPS_PER_FRAME);
        const std::vector<double> sf = encode_chips(frame_index, callsign);
        const std::size_t take = std::min(sf.size(), n_chips - pos);
        std::copy_n(sf.begin(), take, out.begin() + static_cast<std::ptrdiff_t>(pos));
        pos += take;
    }
    return out;
}

std::vector<std::int64_t> find_sync(std::span<const double> chips,
                                    double threshold, int max_candidates) {
    const std::size_t n = chips.size();
    if (n < static_cast<std::size_t>(SYNC_LEN)) return {};
    const std::size_t n_out = n - SYNC_LEN + 1;

    double sync_energy = 0.0;
    for (int i = 0; i < SYNC_LEN; ++i) {
        const double s = static_cast<double>(config::BEACON_SYNC[static_cast<std::size_t>(i)]);
        sync_energy += s * s;
    }

    std::vector<double> score(n_out);
    for (std::size_t k = 0; k < n_out; ++k) {
        // np.correlate(chips, SYNC, "valid"): no reversal.
        double corr = 0.0;
        double energy = 0.0;
        for (int j = 0; j < SYNC_LEN; ++j) {
            const double c = chips[k + static_cast<std::size_t>(j)];
            corr += c * static_cast<double>(
                            config::BEACON_SYNC[static_cast<std::size_t>(j)]);
            energy += c * c;
        }
        score[k] = corr / (std::sqrt(energy * sync_energy) + 1e-12);
    }

    // Best first. Ties break toward the lower offset, which numpy's
    // argsort does not promise -- it defaults to an unstable sort. Exact
    // ties between float scores are vanishingly unlikely here, and a tie
    // that mattered would mean two candidates both passing CRC, i.e. a
    // genuine ambiguity rather than a port difference.
    std::vector<std::size_t> order(n_out);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&score](std::size_t a, std::size_t b) {
                         return score[a] > score[b];
                     });

    std::vector<std::int64_t> out;
    const std::size_t limit =
        std::min(order.size(), static_cast<std::size_t>(max_candidates) * 4);
    for (std::size_t i = 0; i < limit; ++i) {
        if (score[order[i]] >= threshold)
            out.push_back(static_cast<std::int64_t>(order[i]));
        if (out.size() >= static_cast<std::size_t>(max_candidates)) break;
    }
    return out;
}

namespace {

std::optional<std::pair<int, std::string>> decode_payload(
    std::span<const double> coded_chips) {
    std::vector<int> padded;
    padded.reserve(PADDED_PAYLOAD_BITS);
    for (int i = 0; i < N_CHUNKS; ++i) {
        const int data = golay::decode_soft(
            coded_chips.subspan(static_cast<std::size_t>(i) * 24, 24));
        append_int_bits(padded, data, 12);
    }
    const std::span<const int> payload(padded.data(), PAYLOAD_BITS);
    const std::span<const int> body = payload.subspan(
        0, static_cast<std::size_t>(PAYLOAD_BITS - BEACON_CRC_BITS));
    const std::span<const int> crc_bits = payload.subspan(
        static_cast<std::size_t>(PAYLOAD_BITS - BEACON_CRC_BITS));

    const std::vector<int> want = crc16(body);
    if (!std::equal(want.begin(), want.end(), crc_bits.begin())) return std::nullopt;

    const int frame_index = bits_to_int(body.subspan(0, BEACON_COUNTER_BITS));
    std::vector<int> codes;
    for (int i = 0; i < config::BEACON_CALLSIGN_BITS;
         i += BEACON_CALLSIGN_CHAR_BITS)
        codes.push_back(bits_to_int(body.subspan(
            static_cast<std::size_t>(BEACON_COUNTER_BITS + i),
            BEACON_CALLSIGN_CHAR_BITS)));
    return std::make_pair(frame_index, codes_to_callsign(codes));
}

}  // namespace

std::optional<BeaconResult> decode(std::span<const double> chips,
                                   double threshold) {
    for (std::int64_t off : find_sync(chips, threshold)) {
        const std::size_t end =
            static_cast<std::size_t>(off) + SYNC_LEN + CODED_LEN;
        if (end > chips.size()) continue;
        const auto result = decode_payload(
            chips.subspan(static_cast<std::size_t>(off) + SYNC_LEN, CODED_LEN));
        if (result)
            return BeaconResult{off, result->first, result->second};
    }
    return std::nullopt;
}

}  // namespace sstvae::beacon
