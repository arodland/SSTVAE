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

std::optional<BeaconResult> decode_single_repetition(
    std::int64_t chip_offset, std::span<const double> coded_chips) {
    const auto result = decode_payload(coded_chips);
    if (!result) return std::nullopt;
    return BeaconResult{chip_offset, result->first, result->second};
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
// decode_combined below uses both, in three tiers:
//
// - Chunks entirely inside the callsign field are identical, chip for
//   chip, in every repetition, so their *signs* can simply be summed
//   across repetitions before a single Golay decode -- genuine coherent
//   combining, the cheap case. Sign, not raw magnitude: under fading,
//   the per-frame channel estimate the equalizer divides by can itself
//   sit near a fade null, amplifying that frame's noise into an
//   enormous, essentially random magnitude, so a single such repetition
//   dominates a raw sum and wrecks it (measured in the Python
//   reference: raw summing recovered 6/12 bits -- chance -- where sign
//   summing recovered 12/12).
// - The chunk carrying the counter can't be summed that way (a
//   genuinely different value is correct at every repetition), so
//   search_counter_chunk instead evaluates every hypothesis by
//   regenerating each repetition's expected codeword and summing its
//   sign-correlation, combining evidence across repetitions before
//   choosing -- voting on independent per-repetition decodes was tried
//   first and measured useless at the SNRs this needs to help with (18
//   repetitions, 18 different guesses, no repeat at all).
// - The chunk mixing leftover callsign bits with the per-repetition CRC
//   has the same problem but no simple shift predicts it (CRC depends
//   on the whole payload) -- search_crc_mixed_chunk brute-forces its
//   free bits and asks payload_bits what the CRC-inclusive chunk should
//   look like for each hypothesis at each repetition's own counter.
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

// np.sign: 0.0 at exactly zero, not +1 -- matters here because summed
// signs feed straight into a Golay decode via golay::signs_table()
// lookups, so a stray +1 bias at an exact-zero chip (vanishingly likely
// on real noise, but reachable from a deliberately constructed test
// vector) would be a real, if tiny, divergence from the Python
// reference rather than a harmless rounding difference.
double sign(double x) { return x > 0.0 ? 1.0 : (x < 0.0 ? -1.0 : 0.0); }

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
                sign(chips[static_cast<std::size_t>(pos + SYNC_LEN + clo + c)]);
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

// Joint search for a chunk mixing still-unknown callsign bits with the
// per-repetition CRC (chunk 4 in the current layout) -- see the
// file-level comment above. Brute-forces only this chunk's free
// (non-CRC) bits and, for each hypothesis, derives the exact expected
// per-repetition chunk content from payload_bits itself. Returns the
// chunk's free-bit fragment (`n_free` bits, MSB first), or nullopt if
// the chunk isn't shaped as assumed (free callsign bits followed by CRC
// bits, no counter overlap).
std::optional<std::vector<int>> search_crc_mixed_chunk(
    std::span<const double> chips, const std::vector<std::int64_t>& grid,
    std::int64_t anchor_off, int chunk_idx, int frame_index,
    const std::vector<int>& known_callsign_bits) {
    const auto [blo, bhi] = chunk_bit_range(chunk_idx);
    const int callsign_lo = BEACON_COUNTER_BITS;
    const int callsign_hi = callsign_lo + config::BEACON_CALLSIGN_BITS;
    const int free_lo = std::max(blo, callsign_lo);
    const int free_hi = std::min(bhi, callsign_hi);
    const int n_free = free_hi - free_lo;
    if (free_lo != blo || n_free + (bhi - callsign_hi) != 12) return std::nullopt;

    const std::int64_t anchor_frame = anchor_off / CHIPS_PER_FRAME;
    const int clo = chunk_idx * 24;
    const std::size_t n_grid = grid.size();
    std::vector<double> received(n_grid * 24);
    std::vector<std::int64_t> frame_deltas(n_grid);
    for (std::size_t g = 0; g < n_grid; ++g) {
        const std::int64_t pos = grid[g];
        frame_deltas[g] = pos / CHIPS_PER_FRAME - anchor_frame;
        for (int c = 0; c < 24; ++c)
            received[g * 24 + static_cast<std::size_t>(c)] =
                sign(chips[static_cast<std::size_t>(pos + SYNC_LEN + clo + c)]);
    }
    const std::span<const double> signs = golay::signs_table();

    double best_score = -1e300;
    std::optional<std::vector<int>> best_frag;
    for (int free_val = 0; free_val < (1 << n_free); ++free_val) {
        std::vector<int> frag;
        append_int_bits(frag, free_val, n_free);
        std::vector<int> trial = known_callsign_bits;
        std::copy(frag.begin(), frag.end(), trial.begin() + (free_lo - callsign_lo));
        std::vector<int> codes;
        for (int i = 0; i < config::BEACON_CALLSIGN_BITS; i += BEACON_CALLSIGN_CHAR_BITS)
            codes.push_back(bits_to_int(std::span<const int>(trial).subspan(
                static_cast<std::size_t>(i), BEACON_CALLSIGN_CHAR_BITS)));
        const std::string callsign = codes_to_callsign(codes);

        double score = 0.0;
        for (std::size_t g = 0; g < n_grid; ++g) {
            const std::int64_t fi = frame_index + frame_deltas[g];
            if (fi < 0 || fi > MAX_FRAME_COUNTER) continue;
            std::vector<int> payload = payload_bits(static_cast<int>(fi), callsign);
            std::vector<int> chunk_bits(payload.begin() +
                                            std::min<std::ptrdiff_t>(blo, static_cast<std::ptrdiff_t>(payload.size())),
                                        payload.begin() +
                                            std::min<std::ptrdiff_t>(bhi, static_cast<std::ptrdiff_t>(payload.size())));
            chunk_bits.resize(12, 0);
            const int data12 = bits_to_int(chunk_bits);
            const double* row = signs.data() + static_cast<std::size_t>(data12) * golay::N_BITS;
            for (int c = 0; c < 24; ++c)
                score += row[c] * received[g * 24 + static_cast<std::size_t>(c)];
        }
        if (score > best_score) {
            best_score = score;
            best_frag = frag;
        }
    }
    return best_frag;
}

std::optional<BeaconResult> decode_combined(std::span<const double> chips,
                                            std::int64_t anchor_off) {
    const std::int64_t n = static_cast<std::int64_t>(chips.size());
    const std::vector<std::int64_t> grid = repetition_grid(n, anchor_off);
    if (grid.size() < 3) return std::nullopt;  // too few repetitions for this to mean anything

    const int counter_lo = 0;
    const int callsign_lo = BEACON_COUNTER_BITS;
    const int callsign_hi = BEACON_COUNTER_BITS + config::BEACON_CALLSIGN_BITS;

    std::vector<int> invariant_chunks, variant_chunks;
    for (int c = 0; c < N_CHUNKS; ++c) {
        const auto [clo_bit, chi_bit] = chunk_bit_range(c);
        if (callsign_lo <= clo_bit && chi_bit <= callsign_hi)
            invariant_chunks.push_back(c);
        else if (clo_bit < callsign_hi)
            variant_chunks.push_back(c);
    }

    std::vector<int> callsign_bits(static_cast<std::size_t>(config::BEACON_CALLSIGN_BITS), -1);

    // Coherent case: sum this chunk's coded chips (by sign) across every
    // repetition (identical by construction), decode once.
    for (int c : invariant_chunks) {
        const int clo = c * 24;
        std::vector<double> summed(24, 0.0);
        for (std::int64_t pos : grid)
            for (int j = 0; j < 24; ++j)
                summed[static_cast<std::size_t>(j)] +=
                    sign(chips[static_cast<std::size_t>(pos + SYNC_LEN + clo + j)]);
        const std::vector<int> bits = decode_chunk_bits(summed);
        const auto [blo, bhi] = chunk_bit_range(c);
        std::copy(bits.begin(), bits.end(), callsign_bits.begin() + (blo - callsign_lo));
    }

    // The chunk carrying the counter's own low bit can't be voted on --
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
        const int lo = std::max(blo, callsign_lo);
        const int hi = std::min(bhi, callsign_hi);
        if (lo < hi) {
            std::vector<int> extra_bits;
            append_int_bits(extra_bits, extra_bits_value, n_extra);
            std::copy(extra_bits.begin() + (lo - (bhi - n_extra)), extra_bits.end(),
                     callsign_bits.begin() + (lo - callsign_lo));
        }
    }

    // Remaining variant chunks (the one mixing leftover callsign bits
    // with the per-repetition CRC) can't be voted on either -- joint-
    // search them too.
    for (int c : variant_chunks) {
        if (c == counter_chunk) continue;
        const auto frag =
            search_crc_mixed_chunk(chips, grid, anchor_off, c, frame_index, callsign_bits);
        if (!frag) return std::nullopt;
        const int lo = std::max(chunk_bit_range(c).first, callsign_lo);
        std::copy(frag->begin(), frag->end(), callsign_bits.begin() + (lo - callsign_lo));
    }

    if (std::any_of(callsign_bits.begin(), callsign_bits.end(), [](int b) { return b < 0; }))
        return std::nullopt;  // a fragment nobody resolved (shouldn't happen given the layout)

    std::vector<int> codes;
    for (int i = 0; i < config::BEACON_CALLSIGN_BITS; i += BEACON_CALLSIGN_CHAR_BITS)
        codes.push_back(bits_to_int(std::span<const int>(callsign_bits)
                                        .subspan(static_cast<std::size_t>(i),
                                                BEACON_CALLSIGN_CHAR_BITS)));
    const std::string callsign = codes_to_callsign(codes);

    // Verification: only against chunks nothing above was searched
    // against (the CRC/padding-only ones), and by bit-exact independent
    // decode rather than a correlation threshold -- see the file-level
    // comment for why either shortcut is wrong. If even one repetition's
    // own plain Golay decode of every verify chunk matches exactly what
    // payload_bits says it should be, accept.
    std::vector<int> verify_chunks;
    for (int c = 0; c < N_CHUNKS; ++c)
        if (chunk_bit_range(c).first >= callsign_hi) verify_chunks.push_back(c);
    if (verify_chunks.empty()) return std::nullopt;

    bool verified = false;
    for (std::int64_t pos : grid) {
        const std::int64_t fi =
            frame_index + (pos / CHIPS_PER_FRAME - anchor_off / CHIPS_PER_FRAME);
        if (fi < 0 || fi > MAX_FRAME_COUNTER) continue;
        const std::vector<int> payload = payload_bits(static_cast<int>(fi), callsign);
        bool all_match = true;
        for (int c : verify_chunks) {
            const auto [blo, bhi] = chunk_bit_range(c);
            std::vector<int> want(
                payload.begin() + std::min<std::ptrdiff_t>(blo, static_cast<std::ptrdiff_t>(payload.size())),
                payload.begin() + std::min<std::ptrdiff_t>(bhi, static_cast<std::ptrdiff_t>(payload.size())));
            want.resize(12, 0);
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

    return BeaconResult{anchor_off, frame_index, callsign};
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
