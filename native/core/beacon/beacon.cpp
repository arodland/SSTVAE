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
using config::BEACON_MODE_BITS;
using config::BEACON_RESERVED_BITS;
using config::BEACON_RESERVED_VALUE;
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

std::vector<int> payload_bits(int frame_index, std::string_view callsign,
                              int mode_index) {
    if (frame_index < 0 || frame_index > MAX_FRAME_COUNTER)
        throw std::invalid_argument(
            "beacon: frame_index exceeds the counter range");
    if (mode_index < 0 || mode_index > MAX_MODE_INDEX)
        throw std::invalid_argument(
            "beacon: mode_index exceeds the mode field range");
    std::vector<int> body;
    append_int_bits(body, frame_index, BEACON_COUNTER_BITS);
    for (int code : callsign_to_codes(callsign))
        append_int_bits(body, code, BEACON_CALLSIGN_CHAR_BITS);
    append_int_bits(body, mode_index, BEACON_MODE_BITS);
    append_int_bits(body, BEACON_RESERVED_VALUE, BEACON_RESERVED_BITS);

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

std::vector<double> encode_chips(int frame_index, std::string_view callsign,
                                 int mode_index) {
    // The v4 payload fills its chunks exactly (see the static_assert in
    // the header), so the resize is a no-op kept only as belt and braces.
    std::vector<int> padded = payload_bits(frame_index, callsign, mode_index);
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
                                std::string_view callsign, int mode_index) {
    const std::size_t n_chips =
        static_cast<std::size_t>(n_frames) * CHIPS_PER_FRAME;
    std::vector<double> out(n_chips);
    std::size_t pos = 0;
    while (pos < n_chips) {
        const int frame_index =
            start_frame + static_cast<int>(pos / CHIPS_PER_FRAME);
        const std::vector<double> sf =
            encode_chips(frame_index, callsign, mode_index);
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

struct DecodedPayload {
    int frame_index;
    std::string callsign;
    int mode_index;
};

// The reserved bits are CRC-covered but their *value* is deliberately
// not checked: a future sender that assigns them still decodes here
// (the CRC covers whatever was actually sent), which is what "reserved"
// has to mean for fielded receivers.
std::optional<DecodedPayload> decode_payload(
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
    const int mode_index = bits_to_int(body.subspan(
        static_cast<std::size_t>(BEACON_COUNTER_BITS +
                                 config::BEACON_CALLSIGN_BITS),
        BEACON_MODE_BITS));
    return DecodedPayload{frame_index, codes_to_callsign(codes), mode_index};
}

}  // namespace

std::optional<BeaconResult> decode_single_repetition(
    std::int64_t chip_offset, std::span<const double> coded_chips) {
    const auto result = decode_payload(coded_chips);
    if (!result) return std::nullopt;
    return BeaconResult{chip_offset, result->frame_index, result->callsign,
                        result->mode_index};
}

// --- multi-repetition combining ----------------------------------------------
//
// Port of sstvae/modem/beacon.py's module docstring section of the same
// name -- read that for the full reasoning; this is a condensed version
// alongside the code it explains.
//
// The superframe repeats continuously through the whole transmission
// (~5.2 s apart -- up to ~18 times in a mode C transmission), but
// decode() above only looks at whichever single repetition's own
// Golay+CRC decode happens to succeed, discarding every other
// repetition's evidence. Two things are exactly known about every
// repetition within one transmission: the callsign is identical, and
// the frame counter is an exact, predictable function of chip position
// once any one repetition's value is known (chip_stream() lays
// superframes back to back with no gaps, and the decoder's own chip
// array indexes frames uniformly).
//
// decode_combined below uses both, in two tiers:
//
// - Chunks entirely inside the invariant callsign+mode fields are
//   identical, chip for chip, in every repetition, so they can simply be
//   summed across repetitions before a single Golay decode -- coherent
//   (maximal-ratio) combining, the cheap case. The v4 layout (counter |
//   callsign | mode | reserved | CRC, 84 bits exactly) was chosen so the
//   mode field lands here: chunks 1-4 are pure callsign+mode.
//
//   This summed *signs* until 2026-08-25, correctly for what it was
//   given: the chips then came off the zero-forcing equalizer, so a
//   repetition whose channel estimate sat near a fade null contributed
//   an enormous, essentially random magnitude and dominated a raw sum.
//   The chips are maximal-ratio now, so a faded repetition arrives
//   *small* instead, and the ordering inverts -- measured in the Python
//   reference, mpd at 0 dB: equalized+sign 0.40, equalized+raw 0.15,
//   MRC+sign 0.53, MRC+raw 0.90. The two changes compound and neither
//   is safe without the other.
// - The chunk carrying the counter can't be summed that way (a
//   genuinely different value is correct at every repetition), so
//   search_counter_chunk instead evaluates every hypothesis by
//   regenerating each repetition's expected codeword and summing its
//   correlation, combining evidence across repetitions before
//   choosing -- voting on independent per-repetition decodes was tried
//   first and measured useless at the SNRs this needs to help with (18
//   repetitions, 18 different guesses, no repeat at all).
//
// The pre-v4 layout had a third tier -- a chunk mixing leftover callsign
// bits with the per-repetition CRC, brute-forced by a
// search_crc_mixed_chunk. The v4 layout retired it: with the reserved
// field pinned to a known constant and the CRC pushed to the very end,
// no chunk mixes unknown bits with CRC bits, so chunks 5-6
// (reserved+CRC, CRC) are fully *predictable* from a (counter, callsign,
// mode) hypothesis -- which is exactly what makes them usable as
// verification.
//
// The final verification is the part that is easy to get wrong: it
// must check only chunks that were never used to build the hypothesis
// (the CRC/padding-only chunks) via a *bit-exact* independent decode,
// not a correlation over the whole superframe -- a correlation lets the
// chunks the search already fit dominate the score, which measures
// curve-fit quality rather than correctness, and was measured to
// produce a confident, fully wrong BeaconResult on 40 of 40 pure-noise
// trials before this was caught.

namespace {

std::pair<int, int> chunk_bit_range(int chunk_idx) {
    return {chunk_idx * 12, (chunk_idx + 1) * 12};
}

std::vector<int> decode_chunk_bits(std::span<const double> chip_slice) {
    std::vector<int> out;
    append_int_bits(out, golay::decode_soft(chip_slice), 12);
    return out;
}

// Candidate anchor phases (mod SUPERFRAME_LEN) for the combining
// fallback, found by folding the sync-word correlation across every
// period instead of trusting any single repetition's own (noisy,
// 13-chip) peak -- the same "fold across periods" idea BlindAccumulator
// uses for the pilot, applied here to the Barker-13 word.
std::vector<std::int64_t> folded_sync_phases(std::span<const double> chips,
                                             int max_candidates = 4) {
    if (chips.size() < static_cast<std::size_t>(SYNC_LEN + SUPERFRAME_LEN)) return {};
    const std::size_t n_out = chips.size() - static_cast<std::size_t>(SYNC_LEN) + 1;
    std::vector<double> corr(n_out);
    for (std::size_t k = 0; k < n_out; ++k) {
        double c = 0.0;
        for (int j = 0; j < SYNC_LEN; ++j)
            c += chips[k + static_cast<std::size_t>(j)] *
                 static_cast<double>(config::BEACON_SYNC[static_cast<std::size_t>(j)]);
        corr[k] = c;
    }
    std::vector<double> folded(static_cast<std::size_t>(SUPERFRAME_LEN), 0.0);
    for (std::size_t k = 0; k < n_out; ++k)
        folded[k % static_cast<std::size_t>(SUPERFRAME_LEN)] += corr[k];

    std::vector<std::size_t> order(folded.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&folded](std::size_t a, std::size_t b) {
        return folded[a] > folded[b];
    });
    std::vector<std::int64_t> out;
    const std::size_t limit = std::min(order.size(), static_cast<std::size_t>(max_candidates));
    for (std::size_t i = 0; i < limit; ++i) out.push_back(static_cast<std::int64_t>(order[i]));
    return out;
}

// Every chip offset spaced an exact multiple of SUPERFRAME_LEN from
// `anchor_off` that has a full superframe's worth of chips available.
// Checks each direction's fit explicitly rather than assuming the
// anchor's own position fits -- it doesn't always, e.g. an anchor near
// the very end of a short buffer.
std::vector<std::int64_t> repetition_grid(std::int64_t n_chips, std::int64_t anchor_off) {
    const auto fits = [n_chips](std::int64_t pos) {
        return pos >= 0 && pos + SYNC_LEN + CODED_LEN <= n_chips;
    };
    std::vector<std::int64_t> grid;
    if (fits(anchor_off)) grid.push_back(anchor_off);
    for (std::int64_t k = 1; fits(anchor_off + k * SUPERFRAME_LEN); ++k)
        grid.push_back(anchor_off + k * SUPERFRAME_LEN);
    for (std::int64_t k = -1; fits(anchor_off + k * SUPERFRAME_LEN); --k)
        grid.push_back(anchor_off + k * SUPERFRAME_LEN);
    std::sort(grid.begin(), grid.end());
    return grid;
}

// Joint search over the chunk carrying the counter's own low bit 0 --
// see the file-level comment above for why voting can't be used here.
// Evaluates every possible value of this chunk's remaining
// `12 - n_counter_bits` bits at the anchor, regenerates what each
// repetition's own counter-shifted 12-bit value and Golay codeword
// would be for that hypothesis, and scores each hypothesis by summing
// its predicted codeword's sign-correlation against every repetition's
// actual chips. Returns (counter_at_anchor, extra_bits_value).
std::optional<std::pair<int, int>> search_counter_chunk(
    std::span<const double> chips, const std::vector<std::int64_t>& grid,
    std::int64_t anchor_off, int chunk_idx, int n_counter_bits) {
    const int clo = chunk_idx * 24;
    const std::int64_t anchor_frame = anchor_off / CHIPS_PER_FRAME;
    const std::size_t n_grid = grid.size();

    std::vector<double> received(n_grid * 24);
    std::vector<std::int64_t> frame_deltas(n_grid);
    for (std::size_t g = 0; g < n_grid; ++g) {
        const std::int64_t pos = grid[g];
        frame_deltas[g] = pos / CHIPS_PER_FRAME - anchor_frame;
        for (int c = 0; c < 24; ++c)
            received[g * 24 + static_cast<std::size_t>(c)] =
                chips[static_cast<std::size_t>(pos + SYNC_LEN + clo + c)];
    }

    const int n_extra = 12 - n_counter_bits;
    const int n_counters = 1 << n_counter_bits;
    const int counter_mask = n_counters - 1;
    const std::span<const double> signs = golay::signs_table();

    double best_score = -1e300;
    std::optional<std::pair<int, int>> best;
    for (int extra = 0; extra < (1 << n_extra); ++extra) {
        for (int counter = 0; counter < n_counters; ++counter) {
            double score = 0.0;
            for (std::size_t g = 0; g < n_grid; ++g) {
                const int counter_val = (counter + static_cast<int>(frame_deltas[g])) &
                                        counter_mask;
                const int data12 = (counter_val << n_extra) | extra;
                const double* row = signs.data() +
                                    static_cast<std::size_t>(data12) * golay::N_BITS;
                for (int c = 0; c < 24; ++c)
                    score += row[c] * received[g * 24 + static_cast<std::size_t>(c)];
            }
            if (score > best_score) {
                best_score = score;
                best = std::make_pair(counter, extra);
            }
        }
    }
    return best;
}

std::optional<BeaconResult> decode_combined(std::span<const double> chips,
                                            std::int64_t anchor_off) {
    const std::int64_t n = static_cast<std::int64_t>(chips.size());
    const std::vector<std::int64_t> grid = repetition_grid(n, anchor_off);
    if (grid.size() < 3) return std::nullopt;  // too few repetitions for this to mean anything

    const int counter_lo = 0;
    // The invariant region: every payload bit that is identical in every
    // repetition *and* unknown a priori -- callsign plus mode. (The
    // reserved field is invariant too, but its value is a protocol
    // constant, so it belongs to the verify chunks below rather than to
    // anything that needs recovering.)
    const int inv_lo = BEACON_COUNTER_BITS;
    const int inv_hi = inv_lo + config::BEACON_CALLSIGN_BITS + BEACON_MODE_BITS;
    const int n_inv = inv_hi - inv_lo;

    std::vector<int> invariant_chunks, variant_chunks;
    for (int c = 0; c < N_CHUNKS; ++c) {
        const auto [clo_bit, chi_bit] = chunk_bit_range(c);
        if (inv_lo <= clo_bit && chi_bit <= inv_hi)
            invariant_chunks.push_back(c);
        else if (clo_bit < inv_hi)
            variant_chunks.push_back(c);
    }

    // Recovered callsign+mode bits, indexed relative to inv_lo.
    std::vector<int> inv_bits(static_cast<std::size_t>(n_inv), -1);

    // Coherent case: sum this chunk's coded chips (by sign) across every
    // repetition (identical by construction), decode once.
    for (int c : invariant_chunks) {
        const int clo = c * 24;
        std::vector<double> summed(24, 0.0);
        for (std::int64_t pos : grid)
            for (int j = 0; j < 24; ++j)
                summed[static_cast<std::size_t>(j)] +=
                    chips[static_cast<std::size_t>(pos + SYNC_LEN + clo + j)];
        const std::vector<int> bits = decode_chunk_bits(summed);
        const auto [blo, bhi] = chunk_bit_range(c);
        std::copy(bits.begin(), bits.end(), inv_bits.begin() + (blo - inv_lo));
    }

    // The chunk carrying the counter's own low bit can't be summed --
    // joint-search it.
    const int counter_chunk = *std::find_if(
        variant_chunks.begin(), variant_chunks.end(),
        [counter_lo](int c) { return chunk_bit_range(c).first <= counter_lo; });
    const auto result =
        search_counter_chunk(chips, grid, anchor_off, counter_chunk, BEACON_COUNTER_BITS);
    if (!result) return std::nullopt;
    const auto [frame_index, extra_bits_value] = *result;
    if (frame_index < 0 || frame_index > MAX_FRAME_COUNTER) return std::nullopt;
    const int n_extra = 12 - BEACON_COUNTER_BITS;
    {
        const auto [blo, bhi] = chunk_bit_range(counter_chunk);
        const int lo = std::max(blo, inv_lo);
        const int hi = std::min(bhi, inv_hi);
        if (lo < hi) {
            std::vector<int> extra_bits;
            append_int_bits(extra_bits, extra_bits_value, n_extra);
            std::copy(extra_bits.begin() + (lo - (bhi - n_extra)), extra_bits.end(),
                     inv_bits.begin() + (lo - inv_lo));
        }
    }

    if (std::any_of(inv_bits.begin(), inv_bits.end(), [](int b) { return b < 0; }))
        return std::nullopt;  // a fragment nobody resolved (shouldn't happen given the layout)

    std::vector<int> codes;
    for (int i = 0; i < config::BEACON_CALLSIGN_BITS; i += BEACON_CALLSIGN_CHAR_BITS)
        codes.push_back(bits_to_int(std::span<const int>(inv_bits)
                                        .subspan(static_cast<std::size_t>(i),
                                                BEACON_CALLSIGN_CHAR_BITS)));
    const std::string callsign = codes_to_callsign(codes);
    const int mode_index = bits_to_int(std::span<const int>(inv_bits).subspan(
        static_cast<std::size_t>(config::BEACON_CALLSIGN_BITS), BEACON_MODE_BITS));

    // Verification: only against chunks nothing above was searched
    // against, and by bit-exact independent decode rather than a
    // correlation threshold -- see the file-level comment for why either
    // shortcut is wrong. In the v4 layout the verify chunks are 5 and 6:
    // reserved+CRC and pure CRC, both fully predicted by payload_bits
    // from the (counter, callsign, mode) hypothesis (the reserved field
    // at its protocol constant -- so a future sender that assigns those
    // bits loses only this combining fallback on today's receivers,
    // never the single-shot CRC path). If even one repetition's own
    // plain Golay decode of every verify chunk matches exactly, accept.
    std::vector<int> verify_chunks;
    for (int c = 0; c < N_CHUNKS; ++c)
        if (chunk_bit_range(c).first >= inv_hi) verify_chunks.push_back(c);
    if (verify_chunks.empty()) return std::nullopt;

    bool verified = false;
    for (std::int64_t pos : grid) {
        const std::int64_t fi =
            frame_index + (pos / CHIPS_PER_FRAME - anchor_off / CHIPS_PER_FRAME);
        if (fi < 0 || fi > MAX_FRAME_COUNTER) continue;
        const std::vector<int> payload =
            payload_bits(static_cast<int>(fi), callsign, mode_index);
        bool all_match = true;
        for (int c : verify_chunks) {
            const auto [blo, bhi] = chunk_bit_range(c);
            const std::vector<int> want(payload.begin() + blo, payload.begin() + bhi);
            const std::vector<int> got = decode_chunk_bits(
                chips.subspan(static_cast<std::size_t>(pos + SYNC_LEN + c * 24), 24));
            if (!std::equal(want.begin(), want.end(), got.begin())) {
                all_match = false;
                break;
            }
        }
        if (all_match) {
            verified = true;
            break;
        }
    }
    if (!verified) return std::nullopt;

    return BeaconResult{anchor_off, frame_index, callsign, mode_index};
}

}  // namespace

std::optional<BeaconResult> decode(std::span<const double> chips,
                                   double threshold) {
    const std::vector<std::int64_t> candidates = find_sync(chips, threshold);
    for (std::int64_t off : candidates) {
        const std::size_t end =
            static_cast<std::size_t>(off) + SYNC_LEN + CODED_LEN;
        if (end > chips.size()) continue;
        const auto result = decode_single_repetition(
            off, chips.subspan(static_cast<std::size_t>(off) + SYNC_LEN, CODED_LEN));
        if (result) return result;
    }
    for (std::int64_t off : candidates) {
        const auto result = decode_combined(chips, off);
        if (result) return result;
    }
    for (std::int64_t off : folded_sync_phases(chips)) {
        const auto result = decode_combined(chips, off);
        if (result) return result;
    }
    return std::nullopt;
}

}  // namespace sstvae::beacon
