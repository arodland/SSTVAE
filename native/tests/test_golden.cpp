// Check the C++ core against the committed golden corpus.
//
// The corpus is the Python reference's output at each module boundary
// (tools/gen_golden_vectors.py). This binary and `pytest` read the same
// bytes, so neither side gets to hold its own idea of the right answer.
//
// Tolerances are per-check and each one is justified where it is used.
// The rule: anything that is pure integer or pure sign arithmetic must
// match *exactly*, and only sums of transcendentals get a tolerance.

#include <algorithm>
#include <cstdlib>
#include <string>
#include <vector>

#include "check.hpp"
#include "config.hpp"
#include "beacon/beacon.hpp"
#include "dsp/dsp.hpp"
#include "framing/framing.hpp"
#include "golay/golay.hpp"
#include "ofdm/ofdm.hpp"
#include "sync/sync.hpp"
#include "testing/npy.hpp"

using namespace sstvae;
using sstvae::dsp::cdouble;
using sstvae::testing::load_c16;
using sstvae::testing::load_f8;
using sstvae::testing::load_i8;

namespace {

std::string golden_dir;
// Path to sstvae/modem/interleaver_perms.npy. Passed in rather than
// derived from golden_dir, which would bake in the layout of the repo
// above the corpus and break the moment either moved.
std::string frozen_perms_path;

std::string g(const std::string& name) { return golden_dir + "/" + name + ".npy"; }

// Tolerance for a single phasor, and the reason it is not zero.
//
// **Both** implementations now reduce (n*f) mod FS in exact integer
// arithmetic before calling exp(), so the argument is under one turn on
// each side and identical between them. What is left is only that no
// standard requires exp/sin/cos to be correctly rounded, so two libms
// may differ in the last ulp -- about 2.2e-16 on a unit phasor.
//
// Measured C++ against Python: 9.6e-16 on MOD_MATRIX, 9.4e-16 on
// DEMOD_MATRIX, and exactly 0 on the pilot sequence. 1e-14 is ~10x that,
// which is margin for platforms whose libm rounds differently, not slack
// for the port.
//
// This was 2e-13 while sstvae/modem/ofdm.py still built its phasors on
// an unreduced argument reaching 262 rad: the tolerance then had to
// cover the *reference's* ~3e-14 error, which is a much weaker statement
// about the port. See docs/todo.md, item closed 2026-07-28.
constexpr double PHASOR_TOL = 1e-14;

// Sums of NC or M of those phasors, so the per-term error can
// accumulate across 24 or 160 terms; numpy additionally reaches its sums
// through BLAS, which blocks and vectorizes and therefore associates
// differently. Measured worst case 4.9e-15 (modulate_symbols); 1e-13
// gives ~20x. For scale, 1e-13 on a unit phasor is -260 dB.
constexpr double PHASOR_SUM_TOL = 1e-13;

void test_golay() {
    check::equal(golay::min_distance(), 8, "golay/min_distance");

    const std::vector<std::int64_t> codewords = load_i8(g("golay/all_codewords"));
    check::equal(codewords.size(), std::size_t{4096}, "golay/all_codewords size");
    bool all_exact = true;
    for (int m = 0; m < golay::N_MESSAGES; ++m)
        if (static_cast<std::int64_t>(golay::encode(m)) != codewords[static_cast<std::size_t>(m)])
            all_exact = false;
    // Integer arithmetic: exact or broken, no tolerance is meaningful.
    check::is_true(all_exact, "golay/encode matches every reference codeword");

    const std::vector<std::int64_t> msgs = load_i8(g("golay/bits_messages"));
    const std::vector<std::int64_t> bits = load_i8(g("golay/bits_expected"));
    bool bits_ok = true;
    for (std::size_t i = 0; i < msgs.size(); ++i) {
        const auto got = golay::codeword_bits(static_cast<int>(msgs[i]));
        for (int b = 0; b < golay::N_BITS; ++b)
            if (got[static_cast<std::size_t>(b)] !=
                bits[i * golay::N_BITS + static_cast<std::size_t>(b)])
                bits_ok = false;
    }
    check::is_true(bits_ok, "golay/codeword_bits matches reference");

    const testing::NpyFile soft_file = testing::read_npy(g("golay/soft_inputs"));
    const std::vector<double> soft = load_f8(g("golay/soft_inputs"));
    const std::vector<std::int64_t> expected = load_i8(g("golay/soft_expected"));
    const std::size_t n_cases = soft_file.rows();
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < n_cases; ++i) {
        const std::span<const double> row(soft.data() + i * golay::N_BITS,
                                          golay::N_BITS);
        if (golay::decode_soft(row) != expected[i]) ++mismatches;
    }
    // Includes deliberately noisy cases where the reference decoder is
    // *wrong*; the port has to be wrong in the same places, or it is
    // not the same decoder.
    check::equal(mismatches, std::size_t{0},
                 "golay/decode_soft agrees on all " + std::to_string(n_cases) +
                     " cases, including the noisy ones");
}

void test_ofdm_tables() {
    // Frequencies are exactly representable small integers; anything
    // other than equality here means a layout bug, not rounding.
    check::close(std::vector<double>(ofdm::carrier_freqs().begin(),
                                     ofdm::carrier_freqs().end()),
                 load_f8(g("ofdm/carrier_freqs")), 0.0, "ofdm/carrier_freqs exact");
    check::close(std::vector<double>(ofdm::baseband_freqs().begin(),
                                     ofdm::baseband_freqs().end()),
                 load_f8(g("ofdm/baseband_freqs")), 0.0, "ofdm/baseband_freqs exact");

    const auto mod = ofdm::mod_matrix();
    check::close(std::vector<ofdm::cdouble>(mod.begin(), mod.end()),
                 load_c16(g("ofdm/mod_matrix")), PHASOR_TOL, "ofdm/mod_matrix");
    const auto demod = ofdm::demod_matrix();
    check::close(std::vector<ofdm::cdouble>(demod.begin(), demod.end()),
                 load_c16(g("ofdm/demod_matrix")), PHASOR_TOL, "ofdm/demod_matrix");

    // Single calls to polar/exp of an identically-grouped argument: the
    // only possible disagreement is libm's last ulp, hence 1e-15 rather
    // than the looser sum tolerance.
    check::close(std::vector<ofdm::cdouble>(ofdm::pilot_sequence().begin(),
                                            ofdm::pilot_sequence().end()),
                 load_c16(g("ofdm/pilot_sequence")), 1e-15, "ofdm/pilot_sequence");

    check::close(ofdm::preamble_waveform(), load_f8(g("ofdm/preamble_waveform")),
                 PHASOR_SUM_TOL, "ofdm/preamble_waveform");
    check::close(ofdm::preamble_template(), load_c16(g("ofdm/preamble_template")),
                 PHASOR_SUM_TOL, "ofdm/preamble_template");
    check::close(ofdm::pilot_template(), load_c16(g("ofdm/pilot_template")),
                 PHASOR_SUM_TOL, "ofdm/pilot_template");
}

void test_ofdm_transforms() {
    const testing::NpyFile in_file = testing::read_npy(g("ofdm/modulate_input"));
    const std::vector<ofdm::cdouble> symbols = load_c16(g("ofdm/modulate_input"));
    const std::size_t n_sym = in_file.rows();
    check::close(ofdm::modulate_symbols(symbols, n_sym),
                 load_f8(g("ofdm/modulate_expected")), PHASOR_SUM_TOL,
                 "ofdm/modulate_symbols");

    const std::vector<ofdm::cdouble> z = load_c16(g("ofdm/demod_baseband"));
    const std::vector<std::int64_t> starts = load_i8(g("ofdm/demod_starts"));
    const std::vector<std::int64_t> backoffs = load_i8(g("ofdm/demod_backoffs"));
    const std::vector<ofdm::cdouble> expected = load_c16(g("ofdm/demod_expected"));
    std::vector<ofdm::cdouble> got;
    got.reserve(starts.size() * config::NC);
    for (std::size_t i = 0; i < starts.size(); ++i) {
        const auto row = ofdm::demod_window(z, static_cast<std::int64_t>(starts[i]),
                                            static_cast<std::int64_t>(backoffs[i]));
        got.insert(got.end(), row.begin(), row.end());
    }
    check::close(got, expected, PHASOR_SUM_TOL,
                 "ofdm/demod_window over every (start, backoff) pair");

    // The zero-padded tail. This is the case that decides what happens
    // at the end of a recording, so it gets its own vector rather than
    // being trusted to the loop above.
    const std::vector<std::int64_t> tail_start = load_i8(g("ofdm/demod_tail_start"));
    const auto tail = ofdm::demod_window(z, static_cast<std::int64_t>(tail_start[0]));
    check::close(std::vector<ofdm::cdouble>(tail.begin(), tail.end()),
                 load_c16(g("ofdm/demod_tail_expected")), PHASOR_SUM_TOL,
                 "ofdm/demod_window past the end of the signal");
}

void test_dsp() {
    // FIR designs. Tolerance because sinc and the Hamming window are
    // transcendental and the scale normalization sums 129/201 terms;
    // 1e-14 is the same one-ulp-of-exp() reasoning as PHASOR_TOL.
    check::close(dsp::firwin_lowpass(129, 850.0), load_f8(g("dsp/firwin_sync")),
                 1e-14, "dsp/firwin lowpass matches scipy");
    check::close(dsp::firwin_bandpass(201, config::TX_BANDPASS_LO,
                                      config::TX_BANDPASS_HI),
                 load_f8(g("dsp/firwin_tx")), 1e-14,
                 "dsp/firwin bandpass matches scipy");

    const std::vector<double> x = load_f8(g("dsp/signal_input"));

    // Both sides reduce the heterodyne exactly, and it is periodic in 16
    // samples, so this should be as close as two exp() calls can be.
    check::close(dsp::to_baseband(x), load_c16(g("dsp/to_baseband")), 1e-14,
                 "dsp/to_baseband");

    // The FFT is the one place where the two implementations run
    // genuinely different code: SciPy is on ducc0, this is pocketfft.
    // Same lineage, no guarantee of identical bits, and an FFT sums
    // 4096 terms -- hence the looser bound. Still ~1e10 tighter than
    // anything that could affect a decode.
    check::close(dsp::hilbert(x), load_c16(g("dsp/hilbert")), 1e-11,
                 "dsp/hilbert");
    const std::vector<double> x_odd = load_f8(g("dsp/signal_input_odd"));
    check::close(dsp::hilbert(x_odd), load_c16(g("dsp/hilbert_odd")), 1e-11,
                 "dsp/hilbert at odd length (the other mask branch)");

    // FFT-based here (pocketfft) against a direct-sum-equivalent Python
    // reference (scipy.signal.fftconvolve, itself within 1e-15 of the
    // np.convolve values the golden vector was generated from) -- same
    // "FFT sums 4096 terms" cross-implementation spread as dsp/hilbert
    // just above, hence the same bound.
    check::close(dsp::sync_lowpass(dsp::to_baseband(x)),
                 load_c16(g("dsp/sync_lowpass")), 1e-11, "dsp/sync_lowpass");

    const std::vector<double> papr = load_f8(g("dsp/papr_db"));
    check::close(std::vector<double>{dsp::papr_db(x)}, papr, 1e-11, "dsp/papr_db");

    // tx_condition runs config::CLIP_PASSES clip-and-filter passes over a
    // hilbert each, so the FFT difference compounds; it is also the function
    // whose output goes on air, so it gets its own check rather than
    // being trusted to its parts.
    check::close(dsp::tx_condition(x, config::CLIP_HEADROOM_DB),
                 load_f8(g("dsp/tx_condition")), 1e-10, "dsp/tx_condition");

    // Integer output: exact, and specifically checking that half-to-even
    // rounding was used. std::round would differ here.
    const std::vector<std::int64_t> want_i16 = load_i8(g("dsp/to_int16"));
    const std::vector<std::int16_t> got_i16 = dsp::to_int16(x);
    bool i16_ok = got_i16.size() == want_i16.size();
    if (i16_ok)
        for (std::size_t i = 0; i < got_i16.size(); ++i)
            if (static_cast<std::int64_t>(got_i16[i]) != want_i16[i]) i16_ok = false;
    check::is_true(i16_ok, "dsp/to_int16 matches np.round exactly");

    const std::vector<double> offsets = load_f8(g("dsp/freq_correct_offsets"));
    const std::vector<cdouble> fc_expected = load_c16(g("dsp/freq_correct"));
    const std::vector<cdouble> z = dsp::to_baseband(x);
    std::vector<cdouble> fc_got;
    fc_got.reserve(offsets.size() * z.size());
    for (double f : offsets) {
        const auto row = dsp::freq_correct(z, f);
        fc_got.insert(fc_got.end(), row.begin(), row.end());
    }
    check::close(fc_got, fc_expected, 1e-13,
                 "dsp/freq_correct across the acquisition range");
}

void test_framing() {
    using framing::ModeSpec;

    // --- the embedded permutation is the frozen one -------------------
    //
    // Read from sstvae/modem/interleaver_perms.npy directly rather than
    // from a copy in the corpus: that file *is* the on-air format, and a
    // second copy could drift from it. This is the check that makes the
    // property-based tests below sufficient -- with the table verified,
    // interleave and deinterleave are pure index arithmetic.
    const std::string& frozen = frozen_perms_path;
    const testing::NpyFile perms_file = testing::read_npy(frozen);
    const std::vector<std::uint16_t> frozen_perms = testing::load_u2(frozen);
    check::equal(perms_file.rows(), static_cast<std::size_t>(framing::N_GROUPS),
                 "framing/frozen perms group count");
    check::equal(perms_file.cols(), static_cast<std::size_t>(framing::TX_PERM_LEN),
                 "framing/frozen perms length");
    bool table_exact = frozen_perms.size() ==
                       static_cast<std::size_t>(framing::N_GROUPS) * framing::TX_PERM_LEN;
    if (table_exact)
        for (std::size_t i = 0; i < frozen_perms.size(); ++i)
            if (framing::TX_PERMS_DATA[i] != frozen_perms[i]) table_exact = false;
    check::is_true(table_exact,
                   "framing/embedded table equals sstvae/modem/interleaver_perms.npy");

    // Each group's permutation is a valid prefix: distinct indices, all
    // inside the group. True whatever the values are.
    for (int gi = 0; gi < framing::N_GROUPS; ++gi) {
        const auto perm = framing::tx_perm(gi);
        std::vector<bool> seen(config::GROUP_LATENTS, false);
        bool ok = true;
        for (std::uint16_t v : perm) {
            if (v >= config::GROUP_LATENTS || seen[v]) ok = false;
            else seen[v] = true;
        }
        check::is_true(ok, "framing/group " + std::to_string(gi) +
                               " permutation is distinct and in range");
    }

    // --- interleave / deinterleave ------------------------------------
    //
    // Property-based, over mode C so all three groups and the largest
    // group offsets are exercised. The offsets are the subtle part: the
    // table is uint16 and the offsets reach 105,600, which is exactly
    // where a too-narrow type would silently wrap.
    const ModeSpec& mode_c = config::MODES[2];
    std::vector<double> latents(static_cast<std::size_t>(mode_c.n_latents));
    for (std::size_t i = 0; i < latents.size(); ++i)
        latents[i] = static_cast<double>(i);  // exact in double; identity-like
    const std::vector<double> slots = framing::interleave(latents, mode_c);
    check::equal(slots.size(), static_cast<std::size_t>(mode_c.n_tx_latents),
                 "framing/interleave output length");

    const framing::Deinterleaved back = framing::deinterleave(slots, mode_c);
    std::size_t kept = 0, wrong = 0, dropped = 0;
    for (std::size_t i = 0; i < latents.size(); ++i) {
        if (back.weight[i] == 1.0) {
            ++kept;
            if (back.latents[i] != latents[i]) ++wrong;
        } else {
            ++dropped;
            if (back.latents[i] != 0.0) ++wrong;
        }
    }
    check::equal(wrong, std::size_t{0},
                 "framing/round-trip is exact where weight is 1, zero elsewhere");
    check::equal(dropped,
                 static_cast<std::size_t>(mode_c.groups *
                                          config::DROPPED_LATENTS_PER_GROUP),
                 "framing/dropped count matches the documented accounting");
    check::equal(kept, static_cast<std::size_t>(mode_c.n_tx_latents),
                 "framing/kept count matches the transmit budget");

    // Every slot must come from its own group -- the check that would
    // catch a group offset applied wrongly.
    bool groups_ok = true;
    for (int gi = 0; gi < mode_c.groups; ++gi) {
        const std::size_t lo = static_cast<std::size_t>(gi) * config::GROUP_LATENTS;
        for (int i = 0; i < config::TRANSMIT_LATENTS_PER_GROUP; ++i) {
            const double v = slots[static_cast<std::size_t>(gi) *
                                       config::TRANSMIT_LATENTS_PER_GROUP +
                                   static_cast<std::size_t>(i)];
            if (v < static_cast<double>(lo) ||
                v >= static_cast<double>(lo + config::GROUP_LATENTS))
                groups_ok = false;
        }
    }
    check::is_true(groups_ok, "framing/each group's slots stay within that group");

    // --- slot_range_for_frame -----------------------------------------
    const std::vector<std::int64_t> frames = load_i8(g("framing/slot_range_frames"));
    const std::vector<std::int64_t> want_groups =
        load_i8(g("framing/slot_range_groups"));
    const std::vector<std::int64_t> want_idx =
        load_i8(g("framing/slot_range_indices"));
    bool slot_range_ok = true;
    for (std::size_t i = 0; i < frames.size(); ++i) {
        const auto got = framing::slot_range_for_frame(static_cast<int>(frames[i]));
        if (got.group != static_cast<int>(want_groups[i])) slot_range_ok = false;
        for (int j = 0; j < config::LATENTS_PER_FRAME; ++j)
            if (got.indices[static_cast<std::size_t>(j)] !=
                want_idx[i * config::LATENTS_PER_FRAME + static_cast<std::size_t>(j)])
                slot_range_ok = false;
    }
    check::is_true(slot_range_ok,
                   "framing/slot_range_for_frame across every group boundary");

    // --- slots <-> symbols --------------------------------------------
    const std::vector<double> frame_slots = load_f8(g("framing/frame_slots"));
    const std::vector<cdouble> sym = framing::slots_to_symbols(frame_slots);
    check::close(sym, load_c16(g("framing/frame_symbols")), 1e-15,
                 "framing/slots_to_symbols");
    check::close(framing::symbols_to_slots(sym),
                 load_f8(g("framing/frame_slots_roundtrip")), 1e-15,
                 "framing/symbols_to_slots");

    // --- header --------------------------------------------------------
    for (const ModeSpec& mode : config::MODES) {
        const std::string suffix(mode.name);
        const std::vector<std::int64_t> want_bits =
            load_i8(g("framing/header_bits_" + suffix));
        const std::vector<int> got_bits = framing::header_bits(mode);
        bool bits_ok = got_bits.size() == want_bits.size();
        if (bits_ok)
            for (std::size_t i = 0; i < got_bits.size(); ++i)
                if (got_bits[i] != want_bits[i]) bits_ok = false;
        check::is_true(bits_ok, "framing/header_bits mode " + suffix);

        check::close(framing::header_symbol(mode),
                     load_c16(g("framing/header_symbol_" + suffix)), 0.0,
                     "framing/header_symbol mode " + suffix + " (exact: +/-1)");
    }

    // decode_header, including the inputs it must reject. A port that
    // accepted a corrupt header would report a plausible mode and then
    // decode noise, which is worse than reporting no lock at all.
    const testing::NpyFile hdr_file = testing::read_npy(g("framing/header_soft_inputs"));
    const std::vector<double> hdr_soft = load_f8(g("framing/header_soft_inputs"));
    const std::vector<std::int64_t> hdr_want =
        load_i8(g("framing/header_soft_expected"));
    std::size_t hdr_wrong = 0, rejected = 0;
    for (std::size_t i = 0; i < hdr_file.rows(); ++i) {
        const std::span<const double> row(hdr_soft.data() + i * config::NC,
                                          config::NC);
        const auto got = framing::decode_header(row);
        const std::int64_t got_idx = got ? got->index : -1;
        if (got_idx != hdr_want[i]) ++hdr_wrong;
        if (hdr_want[i] == -1) ++rejected;
    }
    check::equal(hdr_wrong, std::size_t{0},
                 "framing/decode_header agrees on all " +
                     std::to_string(hdr_file.rows()) + " cases");
    check::is_true(rejected > 0,
                   "framing/the header cases include some that must be rejected");
}

void test_beacon() {
    check::equal(beacon::SUPERFRAME_LEN, 181, "beacon/superframe length");
    check::equal(beacon::MIN_FRAMES_FOR_SYNC, 73, "beacon/min frames for sync");

    // The 64-symbol alphabet, one code point at a time. The C++ keeps
    // its own copy of the string, so this is what stops the two drifting.
    const std::vector<std::int64_t> alpha_chars =
        load_i8(g("beacon/alphabet_chars"));
    bool alphabet_ok = alpha_chars.size() == beacon::ALPHABET.size();
    if (alphabet_ok)
        for (std::size_t i = 0; i < alpha_chars.size(); ++i)
            if (static_cast<std::int64_t>(
                    static_cast<unsigned char>(beacon::ALPHABET[i])) != alpha_chars[i])
                alphabet_ok = false;
    check::is_true(alphabet_ok, "beacon/alphabet matches the reference exactly");

    // Callsigns, including truncation, padding and characters outside
    // the alphabet (which must become spaces rather than anything else).
    const std::vector<std::int64_t> lengths = load_i8(g("beacon/callsign_lengths"));
    const std::vector<std::int64_t> chars = load_i8(g("beacon/callsign_chars"));
    const std::vector<std::int64_t> codes = load_i8(g("beacon/callsign_codes"));
    std::size_t cpos = 0;
    bool callsigns_ok = true;
    for (std::size_t i = 0; i < lengths.size(); ++i) {
        std::string s;
        for (std::int64_t j = 0; j < lengths[i]; ++j)
            s.push_back(static_cast<char>(chars[cpos++]));
        const std::vector<int> got = beacon::callsign_to_codes(s);
        for (int k = 0; k < config::BEACON_CALLSIGN_CHARS; ++k)
            if (got[static_cast<std::size_t>(k)] !=
                codes[i * config::BEACON_CALLSIGN_CHARS + static_cast<std::size_t>(k)])
                callsigns_ok = false;
    }
    check::is_true(callsigns_ok, "beacon/callsign_to_codes over every case");

    // CRC-16. Integer arithmetic, so exact; the all-zero and all-one
    // inputs are the ones that catch a mis-transcribed shift.
    const std::vector<std::int64_t> crc_lengths =
        load_i8(g("beacon/crc_input_lengths"));
    const std::vector<std::int64_t> crc_inputs = load_i8(g("beacon/crc_inputs"));
    const std::vector<std::int64_t> crc_expected = load_i8(g("beacon/crc_expected"));
    std::size_t bpos = 0;
    bool crc_ok = true;
    for (std::size_t i = 0; i < crc_lengths.size(); ++i) {
        std::vector<int> bits;
        for (std::int64_t j = 0; j < crc_lengths[i]; ++j)
            bits.push_back(static_cast<int>(crc_inputs[bpos++]));
        const std::vector<int> got = beacon::crc16(bits);
        for (int k = 0; k < config::BEACON_CRC_BITS; ++k)
            if (got[static_cast<std::size_t>(k)] !=
                crc_expected[i * config::BEACON_CRC_BITS + static_cast<std::size_t>(k)])
                crc_ok = false;
    }
    check::is_true(crc_ok, "beacon/crc16 over every case");

    // Superframes, at the counter's edges as well as ordinary values,
    // with the mode field cycling through all four values (the unassigned
    // index 3 included -- it must encode, for forward compat).
    // Chips are +/-1 exactly, so no tolerance is defensible.
    const std::vector<std::int64_t> frames = load_i8(g("beacon/encode_frames"));
    const std::vector<std::int64_t> enc_modes = load_i8(g("beacon/encode_modes"));
    const std::vector<double> want_chips = load_f8(g("beacon/encode_chips"));
    std::vector<double> got_chips;
    for (std::size_t i = 0; i < frames.size(); ++i) {
        const auto sf = beacon::encode_chips(static_cast<int>(frames[i]), "KC2G",
                                             static_cast<int>(enc_modes[i]));
        got_chips.insert(got_chips.end(), sf.begin(), sf.end());
    }
    check::close(got_chips, want_chips, 0.0,
                 "beacon/encode_chips is exact (+/-1 chips)");

    // Mode index 1 (B) -- the generator hardcodes the same value.
    const std::vector<double> stream = load_f8(g("beacon/chip_stream"));
    check::close(beacon::chip_stream(0, 120, "N6MTS", 1), stream, 0.0,
                 "beacon/chip_stream is exact");

    // find_sync's ranking, which depends on the normalization as much as
    // on the correlation.
    const std::vector<std::int64_t> want_sync = load_i8(g("beacon/find_sync_offsets"));
    const std::vector<double> head(stream.begin(), stream.begin() + 600);
    const std::vector<std::int64_t> got_sync = beacon::find_sync(head);
    bool sync_ok = got_sync.size() == want_sync.size();
    if (sync_ok)
        for (std::size_t i = 0; i < got_sync.size(); ++i)
            if (got_sync[i] != want_sync[i]) sync_ok = false;
    check::is_true(sync_ok, "beacon/find_sync ranks the same offsets in order");

    // decode(), clean and noisy. The expected values include cases the
    // reference *fails* -- a port that succeeded there would be a
    // different receiver, not a better one.
    const std::vector<std::int64_t> offsets = load_i8(g("beacon/decode_offsets"));
    auto run_decode = [&offsets](const std::vector<double>& src,
                                 const std::vector<std::int64_t>& want,
                                 const std::string& what) {
        std::size_t wrong = 0, failures = 0;
        for (std::size_t i = 0; i < offsets.size(); ++i) {
            const std::size_t off = static_cast<std::size_t>(offsets[i]);
            const std::size_t len =
                std::min<std::size_t>(2 * beacon::SUPERFRAME_LEN,
                                      src.size() > off ? src.size() - off : 0);
            const std::span<const double> window(src.data() + off, len);
            const auto r = beacon::decode(window);
            const std::int64_t got_off = r ? r->chip_offset : -1;
            const std::int64_t got_idx = r ? r->frame_index : -1;
            const std::int64_t got_mode = r ? r->mode_index : -1;
            if (got_off != want[3 * i] || got_idx != want[3 * i + 1] ||
                got_mode != want[3 * i + 2])
                ++wrong;
            if (want[3 * i] == -1) ++failures;
            if (r && want[3 * i] != -1 && r->callsign.empty()) ++wrong;
        }
        check::equal(wrong, std::size_t{0}, what);
        check::is_true(failures > 0,
                       what + ": the cases include some that must fail");
    };
    run_decode(stream, load_i8(g("beacon/decode_expected")),
               "beacon/decode over a clean stream");
    run_decode(load_f8(g("beacon/noisy_stream")),
               load_i8(g("beacon/noisy_decode_expected")),
               "beacon/decode over a noisy stream");

    // The callsign has to survive, not just the counter.
    const auto clean = beacon::decode(stream);
    check::is_true(clean.has_value() && clean->callsign == "N6MTS",
                   "beacon/decode recovers the callsign");
}

void test_sync() {
    // Acquisition is the riskiest code in the port: a wrong timing index
    // is not a small error but a completely different picture, and a
    // wrong CFO bin is a decode that fails with no indication why.
    //
    // The timing index is therefore compared *exactly*. Only the
    // frequency and the metric get a tolerance, and it is the FFT
    // tolerance -- both sides reach these through convolutions of
    // several thousand points.
    constexpr double SYNC_TOL = 1e-9;

    const std::vector<std::string> names = {"clean", "snr6", "snr0", "offset"};
    for (const std::string& name : names) {
        const std::vector<double> wave = load_f8(g("sync/wave_" + name));
        const std::vector<cdouble> z = dsp::to_baseband(wave);

        const std::vector<double> want_acq = load_f8(g("sync/acquire_" + name));
        std::int64_t got_start = -1;
        double got_f = 0.0, got_metric = 0.0;
        try {
            const auto a = sync::acquire(z);
            got_start = a.preamble_start;
            got_f = a.freq_offset;
            got_metric = a.metric;
        } catch (const sync::SyncError&) {
            // -1 is how the corpus records a refusal.
        }
        check::equal(got_start, static_cast<std::int64_t>(want_acq[0]),
                     "sync/acquire " + name + ": preamble_start (exact)");
        if (want_acq[0] >= 0) {
            check::close(std::vector<double>{got_f}, std::vector<double>{want_acq[1]},
                         SYNC_TOL, "sync/acquire " + name + ": freq_offset");
            check::close(std::vector<double>{got_metric},
                         std::vector<double>{want_acq[2]}, SYNC_TOL,
                         "sync/acquire " + name + ": metric");
        }

        const std::vector<double> want_blind = load_f8(g("sync/blind_" + name));
        std::int64_t blind_start = -1;
        double blind_f = 0.0, blind_metric = 0.0;
        try {
            const auto b = sync::acquire_blind(z);
            blind_start = b.frame_start;
            blind_f = b.freq_offset;
            blind_metric = b.metric;
        } catch (const sync::SyncError&) {
        }
        check::equal(blind_start, static_cast<std::int64_t>(want_blind[0]),
                     "sync/acquire_blind " + name + ": frame_start (exact)");
        if (want_blind[0] >= 0) {
            check::close(std::vector<double>{blind_f},
                         std::vector<double>{want_blind[1]}, SYNC_TOL,
                         "sync/acquire_blind " + name + ": freq_offset");
            check::close(std::vector<double>{blind_metric},
                         std::vector<double>{want_blind[2]}, SYNC_TOL,
                         "sync/acquire_blind " + name + ": metric");
        }
    }

    // Pure noise. Refusing to lock is as much a part of the contract as
    // locking: a receiver that found a preamble here would decode noise
    // into a picture and report success.
    const std::vector<double> noise = load_f8(g("sync/wave_noise"));
    const std::vector<double> want_noise = load_f8(g("sync/acquire_noise"));
    bool refused = false;
    try {
        sync::acquire(dsp::to_baseband(noise));
    } catch (const sync::SyncError&) {
        refused = true;
    }
    check::is_true(want_noise[0] < 0,
                   "sync/the noise case is one the reference refuses");
    check::is_true(refused, "sync/acquire refuses to lock onto pure noise");
}

void test_config_header() {
    // config.hpp is generated, so this is not checking arithmetic -- it
    // is checking that the committed header was generated from the
    // config.py the golden vectors came from.
    check::equal(config::NC, 24, "config/NC");
    check::equal(config::M, config::FS / config::RS, "config/M");
    check::equal(static_cast<int>(config::MODES.size()), 3, "config/mode count");
    check::equal(config::MODES[2].n_frames, 660, "config/mode C frames");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage: %s <golden-dir> [frozen-perms.npy]\n\n"
                     "The golden corpus is generated by "
                     "tools/gen_golden_vectors.py. The second argument is\n"
                     "sstvae/modem/interleaver_perms.npy, the frozen on-air\n"
                     "interleave, which is checked against the table compiled\n"
                     "into the library rather than copied into the corpus.\n",
                     argv[0]);
        return 2;
    }
    golden_dir = argv[1];
    frozen_perms_path = (argc > 2)
                            ? argv[2]
                            : golden_dir + "/../../../sstvae/modem/interleaver_perms.npy";

    try {
        test_config_header();
        test_dsp();
        test_framing();
        test_beacon();
        test_sync();
        test_golay();
        test_ofdm_tables();
        test_ofdm_transforms();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return check::report("golden vectors");
}
